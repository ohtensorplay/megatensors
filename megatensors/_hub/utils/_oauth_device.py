# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device Code OAuth (RFC 8628) for browser-based login, plus access token refresh.

The flow: the client requests a device code, displays a URL and a short user code, the user
authorizes in a browser, and the client polls ``POST {ENDPOINT}/oauth/token`` until a token is
issued. Access tokens may come with a refresh token, used to renew them transparently (see
``utils/_auth.py::get_token``).

This module is self-contained protocol logic: no printing, no persistence. Interactive flows
live in ``_login.py`` (human/library) and ``cli/auth.py`` (machine-readable event stream).
"""

import time
from collections.abc import Callable
from typing import TypeGuard, TypedDict, cast
from urllib.parse import urlsplit

import httpx

from .. import constants
from ..errors import DeviceCodeError, OAuthErrorCode
from ._http import get_session, mega_raise_for_status


_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
_REFRESH_TOKEN_GRANT_TYPE = "refresh_token"
_CLI_OAUTH_SCOPES = (
    "repo:read",
    "repo:write",
    "repo:delete",
    "community:write",
    "jobs:run",
    "inference:run",
    "account:keys",
    "webhooks:manage",
    "openid",
    "profile",
    "offline_access",
)


class DeviceCodeInfo(TypedDict):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str  # falls back to verification_uri if the server omits it
    interval: int
    expires_in: int


class OAuthTokenResponse(TypedDict, total=False):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str
    scope: str
    id_token: str


def request_device_code() -> DeviceCodeInfo:
    """Request a device code from the Hub's OAuth device authorization endpoint.

    The returned dict is normalized: `interval`, `expires_in` and `verification_uri_complete`
    are always set (server values, or sensible defaults).

    Raises:
        [`DeviceCodeError`]: If the request fails.
    """
    try:
        response = get_session().post(
            f"{constants.ENDPOINT}/oauth/device",
            data={
                "client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID,
                "scope": " ".join(_CLI_OAUTH_SCOPES),
            },
            timeout=constants.MEGA_HUB_DOWNLOAD_TIMEOUT,
        )
        mega_raise_for_status(response)
    except Exception as error:
        raise DeviceCodeError(
            "Could not start browser authorization. Check your network connection and try again."
        ) from error
    try:
        info = response.json()
    except Exception as error:
        raise DeviceCodeError("The authentication service returned an invalid response. Please try again.") from error
    if not isinstance(info, dict):
        raise DeviceCodeError("The authentication service returned an invalid response. Please try again.")

    device_code = info.get("device_code")
    user_code = info.get("user_code")
    verification_uri = info.get("verification_uri")
    verification_uri_complete = info.get("verification_uri_complete") or verification_uri
    if (
        not isinstance(device_code, str)
        or not device_code
        or not isinstance(user_code, str)
        or not user_code
        or not _is_http_url(verification_uri)
        or not _is_http_url(verification_uri_complete)
    ):
        raise DeviceCodeError("The authentication service returned an invalid response. Please try again.")
    return DeviceCodeInfo(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=verification_uri_complete,
        interval=_bounded_positive_int(info.get("interval"), default=5, maximum=60),
        expires_in=_bounded_positive_int(info.get("expires_in"), default=900, maximum=3600),
    )


def poll_device_token(
    device_info: DeviceCodeInfo, *, on_pending: Callable[[], None] | None = None
) -> OAuthTokenResponse:
    """Poll the token endpoint until the user authorizes the device.

    Args:
        device_info (`DeviceCodeInfo`):
            The device authorization response from [`request_device_code`].
        on_pending (`Callable`, *optional*):
            Called after each "authorization pending" response (e.g. to print a progress dot).

    Returns:
        `OAuthTokenResponse`: the full token response: `access_token`, and optionally
        `refresh_token` and `expires_in`.

    Raises:
        [`DeviceCodeError`]: If authorization is denied, the device code expires, or the server
            returns an unexpected OAuth error.
    """
    interval = device_info["interval"]
    deadline = time.monotonic() + device_info["expires_in"]
    while time.monotonic() < deadline:
        # Inconclusive responses (network blip, 5xx, gateway error page, rate limiting) must not
        # abort the login: keep polling until the device code expires (RFC 8628 section 3.5).
        # The deadline bounds the total wait even if the endpoint is genuinely broken.
        data = None
        try:
            response = get_session().post(
                f"{constants.ENDPOINT}/oauth/token",
                data={
                    "grant_type": _DEVICE_CODE_GRANT_TYPE,
                    "device_code": device_info["device_code"],
                    "client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID,
                },
                timeout=constants.MEGA_HUB_DOWNLOAD_TIMEOUT,
            )
            if response.status_code < 500:
                parsed = response.json()
                data = parsed if isinstance(parsed, dict) else None
        except (httpx.HTTPError, ValueError):
            pass
        except Exception as error:
            raise DeviceCodeError("Could not complete browser authorization. Please try again.") from error

        if data is not None:
            if isinstance(data.get("access_token"), str) and data["access_token"]:
                return cast(OAuthTokenResponse, data)

            match data.get("error"):
                case None:
                    pass  # JSON without an OAuth `error` field (proxy error page, ...): transient
                case OAuthErrorCode.AUTHORIZATION_PENDING:
                    if on_pending is not None:
                        on_pending()
                case OAuthErrorCode.SLOW_DOWN:
                    interval += 5
                case OAuthErrorCode.EXPIRED_TOKEN:
                    raise DeviceCodeError(
                        "Device code expired. Please try again.", error_code=OAuthErrorCode.EXPIRED_TOKEN
                    )
                case OAuthErrorCode.ACCESS_DENIED:
                    raise DeviceCodeError(
                        "Authorization was denied. Please try again.", error_code=OAuthErrorCode.ACCESS_DENIED
                    )
                case error:
                    raise DeviceCodeError(
                        "Browser authorization failed. Please try again.",
                        error_code=str(error) if error is not None else None,
                    )

        time.sleep(interval)

    raise DeviceCodeError("Device code expired (timeout). Please try again.", error_code=OAuthErrorCode.EXPIRED_TOKEN)


def refresh_access_token(refresh_token: str) -> OAuthTokenResponse:
    """Exchange a refresh token for a new access token.

    Returns:
        `OAuthTokenResponse`: the full token response: `access_token`, and optionally a rotated
        `refresh_token` and `expires_in`.

    Raises:
        [`DeviceCodeError`]: If the server rejects the refresh (`error_code="invalid_grant"` when
            the refresh token is expired or revoked) or returns an unexpected response.
    """
    try:
        response = get_session().post(
            f"{constants.ENDPOINT}/oauth/token",
            data={
                "grant_type": _REFRESH_TOKEN_GRANT_TYPE,
                "refresh_token": refresh_token,
                "client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID,
            },
            # An explicit timeout is critical here: this runs inside `get_token()`, so a hung
            # request would otherwise block every Hub call in the process.
            timeout=constants.MEGA_HUB_DOWNLOAD_TIMEOUT,
        )
    except Exception as error:
        raise DeviceCodeError(
            "Could not refresh the access token. Check your network connection and try again."
        ) from error
    data = _parse_token_response(response)
    if isinstance(data.get("access_token"), str) and data["access_token"]:
        return cast(OAuthTokenResponse, data)
    error = data.get("error") if isinstance(data.get("error"), str) else None
    if error == OAuthErrorCode.INVALID_GRANT:
        message = "Your login has expired. Run `mega auth login --force` to sign in again."
    else:
        message = "Could not refresh the access token. Please sign in again."
    raise DeviceCodeError(
        message,
        error_code=error,
    )


def _parse_token_response(response: httpx.Response) -> dict[str, object]:
    try:
        data = response.json()
    except Exception as error:
        raise DeviceCodeError("The authentication service returned an invalid response. Please try again.") from error
    if not isinstance(data, dict):
        raise DeviceCodeError("The authentication service returned an invalid response. Please try again.")
    return data


def _bounded_positive_int(value: object, *, default: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if 0 < parsed <= maximum else default


def _is_http_url(value: object) -> TypeGuard[str]:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

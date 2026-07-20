# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest import TestCase
from unittest.mock import patch

import httpx

from megatensors._hub.errors import DeviceCodeError
from megatensors._hub.utils import _oauth_device


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def post(self, url: str, *, data: dict[str, str], timeout: float) -> _Response:
        self.calls.append((url, data))
        return next(self.responses)


class DeviceOAuthTests(TestCase):
    def test_active_device_flow_uses_registered_cli_client(self):
        session = _Session([_Response({
            "device_code": "mega_device_test",
            "user_code": "ABCD-1234",
            "verification_uri": "https://hub.test/auth/device",
        })])
        with (
            patch.object(_oauth_device, "get_session", return_value=session),
            patch.object(_oauth_device, "mega_raise_for_status"),
            patch.object(_oauth_device.constants, "ENDPOINT", "https://hub.test"),
            patch.object(_oauth_device.constants, "DEVICE_CODE_OAUTH_CLIENT_ID", "mega-cli"),
        ):
            info = _oauth_device.request_device_code()

        self.assertEqual(session.calls, [("https://hub.test/oauth/device", {
            "client_id": "mega-cli",
            "scope": " ".join(_oauth_device._CLI_OAUTH_SCOPES),
        })])
        self.assertEqual(set(_oauth_device._CLI_OAUTH_SCOPES), {
            "repo:read", "repo:write", "repo:delete", "community:write", "jobs:run",
            "inference:run", "account:keys", "webhooks:manage", "openid", "profile",
            "offline_access",
        })
        self.assertEqual(info["interval"], 5)
        self.assertEqual(info["expires_in"], 900)
        self.assertEqual(info["verification_uri_complete"], "https://hub.test/auth/device")

    def test_active_device_flow_polls_registered_oauth_token(self):
        session = _Session([
            _Response({"error": "authorization_pending"}, 400),
            _Response({
                "access_token": "mega_token",
                "refresh_token": "mega_refresh_token",
                "token_type": "Bearer",
                "scope": "repo:read",
            }),
        ])
        with (
            patch.object(_oauth_device, "get_session", return_value=session),
            patch.object(_oauth_device.time, "sleep"),
            patch.object(_oauth_device.constants, "ENDPOINT", "https://hub.test"),
            patch.object(_oauth_device.constants, "DEVICE_CODE_OAUTH_CLIENT_ID", "mega-cli"),
        ):
            token = _oauth_device.poll_device_token({
                "device_code": "mega_device_test",
                "user_code": "ABCD-1234",
                "verification_uri": "https://hub.test/auth/device",
                "verification_uri_complete": "https://hub.test/auth/device?user_code=ABCD-1234",
                "interval": 0,
                "expires_in": 10,
            })

        self.assertEqual(token["access_token"], "mega_token")
        self.assertEqual(session.calls[-1][1], {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": "mega_device_test",
            "client_id": "mega-cli",
        })

    def test_device_request_error_does_not_expose_endpoint_or_transport_details(self):
        class FailingSession:
            def post(self, *_args, **_kwargs):
                raise httpx.ConnectTimeout(
                    "_ssl.c:999: timed out while connecting to https://internal.test/oauth/device"
                )

        with patch.object(_oauth_device, "get_session", return_value=FailingSession()):
            with self.assertRaises(DeviceCodeError) as raised:
                _oauth_device.request_device_code()

        message = str(raised.exception)
        self.assertEqual(
            message,
            "Could not start browser authorization. Check your network connection and try again.",
        )
        self.assertNotIn("internal.test", message)
        self.assertNotIn("/oauth/device", message)
        self.assertNotIn("_ssl.c", message)

    def test_device_request_rejects_malformed_response_without_exposing_payload(self):
        session = _Session([_Response({"error": "database at private.internal failed"})])
        with (
            patch.object(_oauth_device, "get_session", return_value=session),
            patch.object(_oauth_device, "mega_raise_for_status"),
        ):
            with self.assertRaises(DeviceCodeError) as raised:
                _oauth_device.request_device_code()

        self.assertEqual(
            str(raised.exception),
            "The authentication service returned an invalid response. Please try again.",
        )
        self.assertNotIn("private.internal", str(raised.exception))

    def test_poll_error_does_not_expose_oauth_description(self):
        session = _Session([
            _Response(
                {
                    "error": "server_error",
                    "error_description": "upstream https://private.internal/oauth/token failed",
                },
                400,
            )
        ])
        with patch.object(_oauth_device, "get_session", return_value=session):
            with self.assertRaises(DeviceCodeError) as raised:
                _oauth_device.poll_device_token({
                    "device_code": "mega_device_test",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://hub.test/auth/device",
                    "verification_uri_complete": "https://hub.test/auth/device?user_code=ABCD-1234",
                    "interval": 0,
                    "expires_in": 10,
                })

        self.assertEqual(str(raised.exception), "Browser authorization failed. Please try again.")
        self.assertNotIn("private.internal", str(raised.exception))

    def test_refresh_error_does_not_expose_oauth_description(self):
        session = _Session([
            _Response(
                {
                    "error": "server_error",
                    "error_description": "upstream https://private.internal/oauth/token failed",
                },
                500,
            )
        ])
        with patch.object(_oauth_device, "get_session", return_value=session):
            with self.assertRaises(DeviceCodeError) as raised:
                _oauth_device.refresh_access_token("refresh-token")

        self.assertEqual(str(raised.exception), "Could not refresh the access token. Please sign in again.")
        self.assertNotIn("private.internal", str(raised.exception))

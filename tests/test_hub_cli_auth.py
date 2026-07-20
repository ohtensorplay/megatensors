# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from click.testing import CliRunner

from megatensors.hub import AccountKeyInfo
from megatensors._hub import _login
from megatensors._hub.cli._errors import format_known_exception
from megatensors._hub.cli._output import OutputFormat, out
from megatensors._hub.cli.auth import _MEGA_ASCII, auth_cli, print_login_header
from megatensors._hub.errors import LoginError


def test_mega_ascii_preserves_top_row_alignment():
    assert _MEGA_ASCII == "\n".join(
        (
            "    __  ___ ______ ______ ___",
            "   /  |/  // ____// ____//   |",
            "  / /|_/ // __/  / / __ / /| |",
            " / /  / // /___ / /_/ // ___ |",
            "/_/  /_//_____/ \\____//_/  |_|",
        )
    )


def test_login_header_only_displays_branding(capsys):
    previous_mode = out.mode
    out.set_mode(OutputFormat.human)
    out.reset_banner()
    try:
        print_login_header()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == f"{_MEGA_ASCII}\n\n"
    finally:
        out.set_mode(previous_mode)
        out.reset_banner()


def test_browser_login_opens_complete_url_without_repeating_code(monkeypatch, capsys):
    device_info = {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://hub.test/auth/device",
        "verification_uri_complete": "https://hub.test/auth/device?user_code=ABCD-EFGH",
        "interval": 5,
        "expires_in": 900,
    }
    response = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 900,
    }
    opened_urls = []
    messages = []

    monkeypatch.setattr(_login, "request_device_code", lambda: device_info)
    monkeypatch.setattr(
        _login.webbrowser,
        "open",
        lambda url: opened_urls.append(url) or True,
    )
    monkeypatch.setattr(_login, "poll_device_token", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(_login, "_save_oauth_token", lambda _response: ("private-token-name", "alice"))
    monkeypatch.setattr(_login.logger, "info", messages.append)

    _login._device_code_login()

    captured = capsys.readouterr()
    assert opened_urls == [device_info["verification_uri_complete"]]
    assert captured.out == ""
    assert "hub.test" not in captured.out
    assert "ABCD-EFGH" not in captured.out
    assert "Enter the code" not in captured.out
    assert captured.err == (
        "Opening browser...\n"
        "Waiting for browser authorization...\n"
    )
    assert messages == [
        "Login successful.",
        "This token will be refreshed automatically when it expires.",
    ]


def test_browser_login_prints_code_when_complete_url_is_unavailable(monkeypatch, capsys):
    device_info = {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://hub.test/auth/device",
        "verification_uri_complete": "https://hub.test/auth/device",
        "interval": 5,
        "expires_in": 900,
    }

    monkeypatch.setattr(_login, "request_device_code", lambda: device_info)
    monkeypatch.setattr(_login.webbrowser, "open", lambda _url: False)
    monkeypatch.setattr(
        _login,
        "poll_device_token",
        lambda *_args, **_kwargs: {"access_token": "access-token"},
    )
    monkeypatch.setattr(_login, "_save_oauth_token", lambda _response: ("private-token-name", "alice"))
    monkeypatch.setattr(_login.logger, "info", lambda _message: None)

    _login._device_code_login()

    captured = capsys.readouterr()
    assert "Open this URL in your browser" in captured.out
    assert "Enter the code: ABCD-EFGH" in captured.out


def test_empty_access_token_is_rejected_before_an_http_request(monkeypatch):
    monkeypatch.setattr(
        "megatensors._hub.mega_api.MegaApi",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("HTTP client must not be created")),
    )

    with pytest.raises(ValueError, match="Access token cannot be empty"):
        _login._validate_and_save_token("  ", add_to_git_credential=False)


def test_login_status_error_does_not_expose_auth_provider_details(monkeypatch):
    monkeypatch.setattr(
        _login,
        "get_token",
        lambda: (_ for _ in ()).throw(
            RuntimeError("OIDC request to https://private.internal/oauth/token failed")
        ),
    )

    with pytest.raises(LoginError) as raised:
        _login._is_logged_in()

    assert str(raised.value) == "Could not check existing login credentials. Please try again."
    assert "private.internal" not in str(raised.value)


def test_token_validation_error_does_not_expose_endpoint_details(monkeypatch):
    class FailingApi:
        def __init__(self, **_kwargs):
            pass

        def whoami(self, _token):
            raise httpx.ConnectTimeout(
                "_ssl.c:999: timed out while connecting to https://private.internal/api/whoami"
            )

    monkeypatch.setattr("megatensors._hub.mega_api.MegaApi", FailingApi)

    with pytest.raises(LoginError) as raised:
        _login._validate_and_save_token("access-token", add_to_git_credential=False)

    assert str(raised.value) == (
        "Could not validate the access token. Check your network connection and try again."
    )
    assert "private.internal" not in str(raised.value)
    assert "_ssl.c" not in str(raised.value)


def test_token_storage_error_does_not_expose_local_path(monkeypatch):
    class ValidApi:
        def __init__(self, **_kwargs):
            pass

        def whoami(self, _token):
            return {"name": "alice", "auth": {"accessToken": {}}}

    monkeypatch.setattr("megatensors._hub.mega_api.MegaApi", ValidApi)
    monkeypatch.setattr(
        _login,
        "_save_token",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("/private/cache/stored_tokens: permission denied")),
    )

    with pytest.raises(LoginError) as raised:
        _login._validate_and_save_token("access-token", add_to_git_credential=False)

    assert str(raised.value) == "Could not save login credentials on this machine."
    assert "/private/cache" not in str(raised.value)


def test_oauth_response_validation_does_not_expose_invalid_internal_field():
    with pytest.raises(LoginError) as raised:
        _login._save_oauth_token(
            {
                "access_token": "access-token",
                "expires_in": "https://private.internal/oauth/token",
            }
        )

    assert str(raised.value) == (
        "The authentication service returned an invalid response. Please try again."
    )
    assert "private.internal" not in str(raised.value)


def test_http_transport_error_is_sanitized_for_cli_output():
    error = httpx.LocalProtocolError("Illegal header value b'Bearer '")

    assert format_known_exception(error) == (
        "Could not communicate with MEGA Hub. Check your network connection and try again."
    )


def test_agent_login_does_not_expose_internal_token_name(monkeypatch):
    device_info = {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://hub.test/auth/device",
        "verification_uri_complete": "https://hub.test/auth/device?user_code=ABCD-EFGH",
        "interval": 5,
        "expires_in": 900,
    }

    monkeypatch.setattr("megatensors._hub.cli.auth._is_logged_in", lambda: False)
    monkeypatch.setattr("megatensors._hub.cli.auth.request_device_code", lambda: device_info)
    monkeypatch.setattr(
        "megatensors._hub.cli.auth.poll_device_token",
        lambda _device_info: {"access_token": "access-token"},
    )
    monkeypatch.setattr(
        "megatensors._hub.cli.auth._save_oauth_token",
        lambda _response: ("private-token-name", "alice"),
    )

    previous_mode = out.mode
    try:
        result = CliRunner().invoke(auth_cli, ["login", "--format", "agent"])
    finally:
        out.set_mode(previous_mode)

    assert result.exit_code == 0
    assert "Login successful: logged in as alice." in result.output
    assert "private-token-name" not in result.output


def test_auth_whoami_uses_mega_hub_api(monkeypatch):
    calls = []

    class RecordingApi:
        def __init__(self, *, token):
            calls.append(token)

        def whoami(self):
            return {
                "name": "alice",
                "orgs": [{"name": "mega"}],
                "auth": {"accessToken": {
                    "kind": "oauth", "role": "write", "scopes": ["repo:read", "jobs:run"],
                }},
            }

    monkeypatch.setattr("megatensors._hub.cli.auth.MegaHubClient", RecordingApi)
    monkeypatch.setattr("megatensors._hub.cli.auth.get_token", lambda: "mega-token")
    result = CliRunner().invoke(auth_cli, ["whoami"])

    assert result.exit_code == 0
    assert calls == ["mega-token"]
    assert "alice" in result.output
    assert "oauth" in result.output
    assert "repo:read,jobs:run" in result.output


def test_auth_commands_do_not_expose_endpoint_option():
    runner = CliRunner()

    for command in ("login", "whoami"):
        result = runner.invoke(auth_cli, [command, "--help"])
        assert result.exit_code == 0
        assert "--endpoint" not in result.output


def test_account_key_cli_reuses_the_native_client_and_auto_detects_ssh(
    tmp_path, monkeypatch
):
    public_key = tmp_path / "id_ed25519.pub"
    public_key.write_text("ssh-ed25519 AAAAC3NzaExample laptop\n", encoding="utf-8")
    calls = []

    class RecordingClient:
        def __init__(self, *, token):
            calls.append(("init", token))

        def add_account_key(self, **kwargs):
            calls.append(("add", kwargs))
            return AccountKeyInfo(
                key_id="key-1",
                key_type="ssh",
                name=kwargs["name"],
                public_key=kwargs["public_key"],
                fingerprint="SHA256:test",
                created_at="2026-07-12T00:00:00.000Z",
            )

    monkeypatch.setattr("megatensors._hub.cli.auth.MegaHubClient", RecordingClient)
    result = CliRunner().invoke(
        auth_cli,
        ["keys", "add", str(public_key), "--name", "Work laptop", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("init", None),
        (
            "add",
            {
                "key_type": "ssh",
                "name": "Work laptop",
                "public_key": "ssh-ed25519 AAAAC3NzaExample laptop\n",
            },
        ),
    ]
    assert '"key_id": "key-1"' in result.output
    assert '"fingerprint": "SHA256:test"' in result.output


def test_account_key_cli_refuses_private_key_before_constructing_client(
    tmp_path, monkeypatch
):
    private_key = tmp_path / "id_ed25519"
    private_key.write_text(
        "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n-----END OPENSSH PRIVATE KEY-----\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "megatensors._hub.cli.auth.MegaHubClient",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("client must not receive private key material")
        ),
    )

    result = CliRunner().invoke(auth_cli, ["keys", "add", str(private_key)])

    assert result.exit_code == 1
    assert result.exception is not None
    assert "Private keys are never accepted" in str(result.exception)


def test_account_key_cli_lists_and_removes_keys(monkeypatch):
    calls = []

    class RecordingClient:
        def __init__(self, *, token):
            calls.append(("init", token))

        def list_account_keys(self):
            return [
                AccountKeyInfo(
                    key_id="key-1",
                    key_type="gpg",
                    name="Release signing",
                    public_key="public",
                    fingerprint="A" * 40,
                    created_at="2026-07-12T00:00:00.000Z",
                )
            ]

        def delete_account_key(self, key_id):
            calls.append(("delete", key_id))

    monkeypatch.setattr("megatensors._hub.cli.auth.MegaHubClient", RecordingClient)
    runner = CliRunner()

    listed = runner.invoke(auth_cli, ["keys", "list", "--format", "json"])
    removed = runner.invoke(
        auth_cli, ["keys", "remove", "key-1", "--yes", "--format", "json"]
    )

    assert listed.exit_code == 0, listed.output
    assert '"fingerprint": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"' in listed.output
    assert removed.exit_code == 0, removed.output
    assert calls == [("init", None), ("init", None), ("delete", "key-1")]

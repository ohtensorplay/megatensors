import click
import httpx
import pytest
from click.testing import CliRunner

from megatensors._hub.cli import mega
from megatensors._hub.cli import _cli_utils
from megatensors._hub.cli._output import _MEGA_ASCII
from megatensors._hub.errors import DeviceCodeError


def test_mega_hub_cli_exposes_migrated_commands():
    runner = CliRunner()
    result = runner.invoke(mega.app, ["--help"], prog_name="mega")

    assert result.exit_code == 0
    assert set(mega.app.list_commands(None)) == {
        "auth",
        "buckets",
        "cache",
        "collections",
        "convert",
        "cp",
        "datasets",
        "discussions",
        "download",
        "env",
        "extensions",
        "jobs",
        "models",
        "papers",
        "repos",
        "repo-files",
        "sandbox",
        "sign",
        "skills",
        "snapshot",
        "spaces",
        "sync",
        "update",
        "upload",
        "upload-large-folder",
        "version",
        "webhooks",
    }


def test_mega_entrypoint_is_the_native_registry():
    assert mega.main.__module__ == "megatensors._hub.cli.mega"
    assert mega.app.help == "MEGA Hub CLI"


def test_hf_aligned_renames_delete_old_command_names():
    runner = CliRunner()
    for args in (
        ["model", "--help"],
        ["dataset", "--help"],
        ["space", "--help"],
        ["job", "--help"],
        ["repo", "--help"],
        ["repos", "update", "--help"],
        ["jobs", "schedule", "--help"],
        ["jobs", "info", "--help"],
        ["repos", "copy", "--help"],
        ["repos", "tree", "--help"],
        ["repos", "delete-file", "--help"],
        ["spaces", "files", "--help"],
        ["spaces", "tree", "--help"],
    ):
        result = runner.invoke(mega.app, args, prog_name="mega")
        assert result.exit_code != 0, args


def test_mega_cli_does_not_expose_endpoint_option():
    def assert_no_endpoint(command: click.Command) -> None:
        for parameter in command.params:
            assert "--endpoint" not in getattr(parameter, "opts", ())
        if isinstance(command, click.Group):
            for name in command.list_commands(None):
                child = command.get_command(None, name)
                assert child is not None
                assert_no_endpoint(child)

    assert_no_endpoint(mega.app)


def test_update_check_targets_megatensors_and_writes_hint_to_stderr(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(_cli_utils.constants, "MEGA_HUB_DISABLE_UPDATE_CHECK", False)
    monkeypatch.setattr(
        _cli_utils.constants, "CHECK_FOR_UPDATE_DONE_PATH", str(tmp_path / "checked")
    )
    monkeypatch.setattr(
        _cli_utils.importlib.metadata, "version", lambda package: "1.0.0"
    )
    monkeypatch.setattr(_cli_utils, "_fetch_latest_pypi_version", lambda: "1.1.0")
    monkeypatch.setattr(
        _cli_utils, "_get_mega_update_command", lambda: ["mega", "update"]
    )
    previous_mode = _cli_utils.out.mode
    _cli_utils.out.set_mode(_cli_utils.OutputFormat.agent)

    try:
        _cli_utils._check_cli_update()
    finally:
        _cli_utils.out.set_mode(previous_mode)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Hint: MEGA CLI 1.1.0 is available (current: 1.0.0).\n"
        "Run `mega update` to upgrade.\n"
    )
    assert (tmp_path / "checked").exists()


def test_update_command_updates_only_megatensors(monkeypatch):
    monkeypatch.setattr(_cli_utils, "installation_method", lambda: "pip")

    command = _cli_utils._get_mega_update_command()

    assert command is not None
    assert command[-4:] == ["pip", "install", "-U", "megatensors"]


def test_update_check_never_suggests_a_downgrade(monkeypatch, tmp_path):
    hints: list[str] = []
    monkeypatch.setattr(_cli_utils.constants, "MEGA_HUB_DISABLE_UPDATE_CHECK", False)
    monkeypatch.setattr(
        _cli_utils.constants, "CHECK_FOR_UPDATE_DONE_PATH", str(tmp_path / "checked")
    )
    monkeypatch.setattr(
        _cli_utils.importlib.metadata, "version", lambda package: "2.0.0"
    )
    monkeypatch.setattr(_cli_utils, "_fetch_latest_pypi_version", lambda: "1.9.0")
    monkeypatch.setattr(_cli_utils.out, "hint", hints.append)

    _cli_utils._check_cli_update()

    assert hints == []


def test_quiet_mode_is_detected_before_update_check():
    assert mega._quiet_requested(["repos", "list", "--quiet"])
    assert mega._quiet_requested(["download", "mega/demo", "--format", "quiet"])
    assert mega._quiet_requested(["cache", "list", "--format=quiet"])
    assert not mega._quiet_requested(["download", "mega/demo", "--", "--quiet"])


def test_requested_output_format_is_detected_before_command_dispatch():
    assert (
        mega._requested_output_format(["download", "mega/demo", "--format", "human"])
        == mega.OutputFormat.human
    )
    assert (
        mega._requested_output_format(["download", "mega/demo", "--format=json"])
        == mega.OutputFormat.json
    )
    assert (
        mega._requested_output_format(["download", "mega/demo", "--json"])
        == mega.OutputFormat.json
    )
    assert (
        mega._requested_output_format(["download", "mega/demo", "--format", "agent"])
        == mega.OutputFormat.agent
    )
    assert (
        mega._requested_output_format(
            ["download", "mega/demo", "--", "--format", "json"]
        )
        is None
    )


def test_main_prints_human_banner_to_stderr(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        mega.sys, "argv", ["mega", "download", "mega/demo", "--format", "human"]
    )
    monkeypatch.setattr(mega, "check_cli_update", lambda: calls.append("update"))
    monkeypatch.setattr(mega, "app", lambda: calls.append("app"))
    previous_mode = mega.out.mode
    mega.out.reset_banner()

    try:
        mega.main()
    finally:
        mega.out.set_mode(previous_mode)
        mega.out.reset_banner()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{_MEGA_ASCII}\n\n"
    assert calls == ["update", "app"]


def test_main_skips_banner_for_machine_output(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        mega.sys, "argv", ["mega", "download", "mega/demo", "--format", "json"]
    )
    monkeypatch.setattr(mega, "check_cli_update", lambda: calls.append("update"))
    monkeypatch.setattr(mega, "app", lambda: calls.append("app"))
    previous_mode = mega.out.mode
    mega.out.reset_banner()

    try:
        mega.main()
    finally:
        mega.out.set_mode(previous_mode)
        mega.out.reset_banner()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert calls == ["update", "app"]


def test_main_sanitizes_http_transport_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        mega.sys, "argv", ["mega", "auth", "login", "--format", "agent"]
    )
    monkeypatch.setattr(mega, "check_cli_update", lambda: None)
    monkeypatch.setattr(
        mega,
        "app",
        lambda: (_ for _ in ()).throw(
            httpx.LocalProtocolError("Illegal header value b'Bearer '")
        ),
    )
    monkeypatch.setattr(mega.constants, "MEGA_DEBUG", False)
    previous_mode = mega.out.mode
    mega.out.reset_banner()

    try:
        with pytest.raises(SystemExit) as exit_info:
            mega.main()
    finally:
        mega.out.set_mode(previous_mode)
        mega.out.reset_banner()

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "Could not communicate with MEGA Hub" in captured.err
    assert "Illegal header value" not in captured.err
    assert "Traceback" not in captured.err


def test_main_does_not_expose_device_endpoint_or_root_cause(monkeypatch, capsys):
    raw_error = httpx.ConnectTimeout(
        "_ssl.c:999: timed out while connecting to https://private.internal/oauth/device"
    )
    login_error = DeviceCodeError(
        "Could not start browser authorization. Check your network connection and try again."
    )
    login_error.__cause__ = raw_error
    monkeypatch.setattr(
        mega.sys, "argv", ["mega", "auth", "login", "--format", "agent"]
    )
    monkeypatch.setattr(mega, "check_cli_update", lambda: None)
    monkeypatch.setattr(mega, "app", lambda: (_ for _ in ()).throw(login_error))
    monkeypatch.setattr(mega.constants, "MEGA_DEBUG", False)
    previous_mode = mega.out.mode
    mega.out.reset_banner()

    try:
        with pytest.raises(SystemExit) as exit_info:
            mega.main()
    finally:
        mega.out.set_mode(previous_mode)
        mega.out.reset_banner()

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert "Login failed: Could not start browser authorization." in captured.err
    assert "private.internal" not in captured.err
    assert "/oauth/device" not in captured.err
    assert "_ssl.c" not in captured.err

import json
from datetime import datetime, timezone

from click.testing import CliRunner

from megatensors._hub.cli import extensions


def test_extensions_list_and_dynamic_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(extensions, "EXTENSIONS_ROOT", tmp_path)
    directory = tmp_path / "hello"
    directory.mkdir()
    executable = directory / "mega-hello"
    executable.write_text("#!/bin/sh\nprintf 'extension:%s\\n' \"$1\"\n")
    executable.chmod(0o755)
    manifest = extensions.ExtensionManifest(
        owner="ohtensorplay",
        repo="mega-hello",
        repo_id="ohtensorplay/mega-hello",
        short_name="hello",
        executable_path=str(executable),
        type="binary",
        installed_at=datetime.now(timezone.utc),
    )
    manifest.save(directory)

    listed = CliRunner().invoke(extensions.extensions_cli, ["list", "--format", "json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)[0]["command"] == "mega hello"
    calls = []
    monkeypatch.setattr(extensions, "execute_extension", lambda name, args: calls.append((name, args)) or 0)
    command = extensions.dynamic_extension_command("hello")
    assert command is not None
    executed = CliRunner().invoke(command, ["world"])
    assert executed.exit_code == 0, executed.output
    assert calls == [("hello", ["world"])]


def test_extension_names_require_mega_prefix():
    try:
        extensions._repo("owner/hf-example")
    except Exception as error:
        assert "mega-" in str(error)
    else:
        raise AssertionError("non-MEGA extension name was accepted")

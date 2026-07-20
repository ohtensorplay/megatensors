import base64
import json
import shlex
from datetime import datetime, timezone
from types import SimpleNamespace

from click.testing import CliRunner

from megatensors._hub.cli import models
from megatensors._hub._space_api import SpaceHardware
from megatensors.hub import FileInfo, RepoInfo


def test_typed_repo_groups_share_the_native_list_client(monkeypatch):
    calls: list[tuple[int, str]] = []

    class FakeMegaClient:
        def __init__(self, *, endpoint=None, token=None):
            pass

        def list_repos(self, *, limit, repo_type):
            calls.append((limit, repo_type))
            return [
                RepoInfo(
                    repo_id=f"mega/{repo_type}-demo",
                    repo_type=repo_type,
                    private=False,
                    created_at="created",
                    updated_at="updated",
                )
            ]

    monkeypatch.setattr(models, "MegaHubClient", FakeMegaClient)
    runner = CliRunner()

    for group, repo_type in (
        (models.models_cli, "model"),
        (models.datasets_cli, "dataset"),
        (models.spaces_cli, "space"),
    ):
        result = runner.invoke(group, ["list", "--limit", "3", "--format", "json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == [
            {
                "repo_id": f"mega/{repo_type}-demo",
                "visibility": "public",
                "updated_at": "updated",
            }
        ]

    assert calls == [(3, "model"), (3, "dataset"), (3, "space")]


def test_models_list_repo_replaces_the_old_files_command(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, token=None):
            pass

        def repo_info(self, repo_id):
            return RepoInfo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
                created_at="created",
                updated_at="updated",
            )

        def list_files(self, repo_id, *, revision):
            calls.append((repo_id, revision))
            return [FileInfo(path="config.json", size=2, sha256="a" * 64)]

    monkeypatch.setattr(models, "MegaHubClient", FakeMegaClient)
    result = CliRunner().invoke(
        models.models_cli,
        ["list", "mega/demo", "--revision", "release", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["path"] == "config.json"
    assert calls == [("mega/demo", "release")]


def test_typed_repo_upload_reuses_canonical_upload(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(models, "run_upload", lambda repo_id, **kwargs: calls.append((repo_id, kwargs)))

    result = CliRunner().invoke(
        models.datasets_cli,
        ["upload", "mega/demo", str(tmp_path), "--commit-description", "Dataset release notes"],
    )

    assert result.exit_code == 0, result.output
    assert calls[0][0] == "mega/demo"
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[0][1]["local_path"] == tmp_path
    assert calls[0][1]["commit_description"] == "Dataset release notes"


def test_typed_repo_download_reuses_canonical_download(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(models, "run_download", lambda repo_id, **kwargs: calls.append((repo_id, kwargs)))

    result = CliRunner().invoke(
        models.spaces_cli,
        ["download", "mega/demo", "app.py", "--local-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls[0][0] == "mega/demo"
    assert calls[0][1]["repo_type"] == "space"
    assert calls[0][1]["filenames"] == ["app.py"]
    assert calls[0][1]["local_dir"] == tmp_path


def test_datasets_parquet_matches_hf_cli_shape(monkeypatch):
    calls = []

    class FakeMegaApi:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def list_dataset_parquet_files(self, *, repo_id, config=None):
            calls.append(("parquet", repo_id, config))
            return [
                SimpleNamespace(config="default", split="train", url="https://mega.example/train.parquet", size=12),
                SimpleNamespace(config="default", split="test", url="https://mega.example/test.parquet", size=8),
            ]

    monkeypatch.setattr(models, "MegaApi", FakeMegaApi)
    result = CliRunner().invoke(
        models.datasets_cli,
        ["parquet", "mega/demo", "--subset", "default", "--split", "train", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [{
        "subset": "default", "split": "train", "url": "https://mega.example/train.parquet", "size": 12,
    }]
    assert calls == [("init", None), ("parquet", "mega/demo", "default")]


def test_datasets_sql_uses_local_dataset_viewer_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(models, "execute_raw_sql_query", lambda *, sql_query, token: calls.append((sql_query, token)) or [{"rows": 2}])

    result = CliRunner().invoke(
        models.datasets_cli,
        ["sql", "SELECT 2 AS rows", "--token", "mega_token", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [{"rows": 2}]
    assert calls == [("SELECT 2 AS rows", "mega_token")]


def test_spaces_runtime_commands_use_hf_compatible_api_and_structured_output(monkeypatch):
    calls = []
    runtime = SimpleNamespace(
        stage="RUNNING",
        hardware="cpu-basic",
        requested_hardware="cpu-upgrade",
        sleep_time=300,
        storage=None,
        dev_mode=False,
        volumes=[],
        raw={"generation": 3, "updatedAt": "2026-07-16T00:00:00Z"},
    )

    class FakeSpaceApi:
        def list_spaces_hardware(self):
            calls.append(("hardware_list",))
            return [
                SimpleNamespace(
                    name="cpu-basic",
                    pretty_name="CPU Basic",
                    cpu="2 vCPU",
                    ram="16 GB",
                    ephemeral_storage="20 GB",
                    accelerator=None,
                    unit_cost_usd=0,
                ),
            ]

        def get_space_runtime(self, repo_id):
            calls.append(("runtime", repo_id))
            return runtime

        def pause_space(self, repo_id):
            calls.append(("pause", repo_id))
            return SimpleNamespace(stage="PAUSED")

        def restart_space(self, repo_id, *, factory_reboot):
            calls.append(("restart", repo_id, factory_reboot))
            return SimpleNamespace(stage="APP_STARTING")

        def request_space_hardware(self, repo_id, *, hardware, sleep_time):
            calls.append(("hardware", repo_id, hardware, sleep_time))
            return runtime

    monkeypatch.setattr(models, "_space_api", lambda token: FakeSpaceApi())
    runner = CliRunner()

    runtime_result = runner.invoke(models.spaces_cli, ["runtime", "mega/demo", "--format", "json"])
    hardware_result = runner.invoke(models.spaces_cli, ["hardware", "--format", "json"])
    pause_result = runner.invoke(models.spaces_cli, ["pause", "mega/demo", "--format", "json"])
    restart_result = runner.invoke(
        models.spaces_cli,
        ["restart", "mega/demo", "--factory-reboot", "--format", "json"],
    )
    settings_result = runner.invoke(
        models.spaces_cli,
        ["settings", "mega/demo", "--hardware", "cpu-upgrade", "--sleep-time", "300", "--format", "json"],
    )

    assert runtime_result.exit_code == 0, runtime_result.output
    assert json.loads(runtime_result.output)["stage"] == "RUNNING"
    assert hardware_result.exit_code == 0, hardware_result.output
    assert json.loads(hardware_result.output.splitlines()[0]) == [{
        "name": "cpu-basic",
        "pretty name": "CPU Basic",
        "cpu": "2 vCPU",
        "ram": "16 GB",
        "accelerator": None,
        "cost/min": "free",
        "cost/hour": "free",
    }]
    assert pause_result.exit_code == 0, pause_result.output
    assert json.loads(pause_result.output) == {"space_id": "mega/demo", "stage": "PAUSED"}
    assert restart_result.exit_code == 0, restart_result.output
    assert json.loads(restart_result.output)["factory_reboot"] is True
    assert settings_result.exit_code == 0, settings_result.output
    assert json.loads(settings_result.output.splitlines()[0]) == {
        "space_id": "mega/demo",
        "hardware": "cpu-upgrade",
        "sleep_time": 300,
    }
    assert calls == [
        ("runtime", "mega/demo"),
        ("hardware_list",),
        ("pause", "mega/demo"),
        ("restart", "mega/demo", True),
        ("hardware", "mega/demo", "cpu-upgrade", 300),
    ]


def test_space_hardware_uses_hf_compatibility_identifiers_but_not_as_a_mega_catalogue():
    assert [hardware.value for hardware in SpaceHardware] == [
        "cpu-basic", "cpu-upgrade", "zero-a10g", "t4-small", "t4-medium", "l4x1", "l4x4",
        "l40sx1", "l40sx4", "l40sx8", "a10g-small", "a10g-large", "a10g-largex2",
        "a10g-largex4", "a100-large", "a100x4", "a100x8",
    ]
    assert "gpu-nano" not in {hardware.value for hardware in SpaceHardware}


def test_spaces_ssh_dry_run_uses_access_proxy_and_safe_argv_payload(monkeypatch):
    class FakeSpaceApi:
        def space_info(self, repo_id):
            assert repo_id == "mega/demo"
            return SimpleNamespace(
                runtime=SimpleNamespace(dev_mode=True),
                subdomain="space-demo",
            )

    monkeypatch.setattr(models, "_space_api", lambda token: FakeSpaceApi())
    monkeypatch.setattr(models.shutil, "which", lambda name: "/usr/bin/cloudflared")
    result = CliRunner().invoke(
        models.spaces_cli,
        ["ssh", "mega/demo", "--dry-run", "python", "-V"],
    )

    assert result.exit_code == 0, result.output
    command = shlex.split(result.output.strip())
    assert command[:5] == [
        "ssh", "-o", "ProxyCommand=/usr/bin/cloudflared access ssh --hostname %h",
        "-o", "RequestTTY=force",
    ]
    assert command[-2] == "mega-spaces@ssh.mega.space"
    payload = command[-1] + "=" * (-len(command[-1]) % 4)
    assert json.loads(base64.urlsafe_b64decode(payload)) == {
        "repo_id": "mega/demo",
        "command": ["python", "-V"],
    }


def test_spaces_secrets_and_variables_never_print_secret_values(monkeypatch):
    calls = []

    class FakeSpaceApi:
        def get_space_secrets(self, repo_id):
            return {
                "API_TOKEN": SimpleNamespace(
                    key="API_TOKEN",
                    description="write only",
                    updated_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
                )
            }

        def add_space_secret(self, repo_id, *, key, value):
            calls.append(("secret", repo_id, key, value))

        def add_space_variable(self, repo_id, *, key, value):
            calls.append(("variable", repo_id, key, value))

        def get_space_variables(self, repo_id):
            return {}

    monkeypatch.setattr(models, "_space_api", lambda token: FakeSpaceApi())
    runner = CliRunner()

    secret_result = runner.invoke(
        models.space_secrets_cli,
        ["add", "mega/demo", "-s", "API_TOKEN=super-secret", "--format", "json"],
    )
    list_result = runner.invoke(
        models.space_secrets_cli,
        ["list", "mega/demo", "--format", "json"],
    )
    variable_result = runner.invoke(
        models.space_variables_cli,
        ["add", "mega/demo", "-e", "MODE=production", "--format", "json"],
    )

    assert secret_result.exit_code == 0, secret_result.output
    assert "super-secret" not in secret_result.output
    assert json.loads(list_result.output)[0]["key"] == "API_TOKEN"
    assert variable_result.exit_code == 0, variable_result.output
    assert calls == [
        ("secret", "mega/demo", "API_TOKEN", "super-secret"),
        ("variable", "mega/demo", "MODE", "production"),
    ]

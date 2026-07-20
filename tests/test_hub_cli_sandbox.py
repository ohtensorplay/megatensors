from types import SimpleNamespace

from click.testing import CliRunner

from megatensors._hub.cli import sandbox


def test_sandbox_create_uses_native_session_api(monkeypatch):
    calls = []

    class FakeSandbox:
        id = "sandbox-0123456789abcdef0123456789abcdef"
        image = "python-3.13"

        @classmethod
        def create(cls, **kwargs):
            calls.append(kwargs)
            return cls()

        def close(self):
            pass

    monkeypatch.setattr(sandbox, "Sandbox", FakeSandbox)
    result = CliRunner().invoke(
        sandbox.sandbox_cli,
        ["create", "--env", "MODE=test", "--secrets", "API_TOKEN=secret", "--allow-egress", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert "secret" not in result.output
    assert calls == [{
        "runtime": "python-3.13",
        "flavor": "cpu-basic",
        "idle_timeout": sandbox.DEFAULT_IDLE_TIMEOUT,
        "max_lifetime": "1h",
        "env": {"MODE": "test"},
        "secrets": {"API_TOKEN": "secret"},
        "namespace": None,
        "allow_egress": True,
        "volumes": [],
        "token": None,
    }]


def test_sandbox_command_tree_matches_hf_names():
    runner = CliRunner()
    result = runner.invoke(sandbox.sandbox_cli, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("create", "exec", "spawn", "cp", "kill", "pool", "process"):
        assert name in result.output


def test_sandbox_pool_create_uses_the_native_fixed_broker_boundary(monkeypatch):
    calls = []

    class FakePool:
        name = "pool-0123456789abcdef0123456789abcdef"
        image = "python:3.13"
        flavor = "cpu-basic"
        host_ids = [name]

        @classmethod
        def create_pool(cls, **kwargs):
            calls.append(kwargs)
            return cls()

    monkeypatch.setattr(sandbox, "SandboxPool", FakePool)
    result = CliRunner().invoke(sandbox.sandbox_cli, ["pool", "create", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert calls == [{
        "image": "python:3.13",
        "flavor": "cpu-basic",
        "per_host": 4,
        "max_hosts": None,
        "idle_timeout": sandbox.DEFAULT_IDLE_TIMEOUT,
        "namespace": None,
        "token": None,
    }]

from __future__ import annotations

import base64
import json
import shlex
from types import SimpleNamespace

from click.testing import CliRunner

from megatensors._hub.cli import job_schedules, job_uv, jobs
from megatensors._hub._space_api import Volume


def _job(stage: str = "SCHEDULING"):
    return SimpleNamespace(
        id="job-0123456789abcdef0123456789abcdef",
        docker_image="python:3.12-slim",
        command=["python", "-c", "print('ok')"],
        arguments=[],
        owner=SimpleNamespace(name="alice"),
        created_at=None,
        started_at=None,
        finished_at=None,
        durations=None,
        labels={},
        billing=None,
        status=SimpleNamespace(stage=stage, message=None),
        url="https://hub.example.test/settings/jobs?job=job-0123456789abcdef0123456789abcdef",
    )


def _schedule():
    return SimpleNamespace(
        id="schedule-0123456789abcdef0123456789abcdef",
        schedule="0 * * * *",
        suspend=False,
        concurrency=False,
        owner=SimpleNamespace(name="alice"),
        created_at=None,
        job_spec=SimpleNamespace(docker_image="python:3.12-slim", command=["python"], arguments=[], labels={}),
        status=SimpleNamespace(last_job=None, next_job_run_at=None),
    )


def test_jobs_run_passes_env_and_reads_secrets_from_local_environment(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def run_job(self, **kwargs):
            calls.append(("run_job", kwargs))
            return _job()

    monkeypatch.setenv("MEGA_JOB_TOKEN", "secret-value")
    monkeypatch.setattr(jobs, "MegaHubClient", FakeClient)
    result = CliRunner().invoke(
        jobs.jobs_cli,
        [
            "run",
            "--detach",
            "--env",
            "MODE=contract",
            "--secret",
            "MEGA_JOB_TOKEN",
            "--volume",
            "mega://buckets/alice/artifacts:/output:rw",
            "--volume",
            "mega://datasets/alice/training@v1/train:/input:ro",
            "python:3.12-slim",
            "python",
            "-c",
            "print('ok')",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[-1] == (
        "run_job",
        {
            "image": "python:3.12-slim",
            "command": ["python", "-c", "print('ok')"],
            "env": {"MODE": "contract"},
            "secrets": {"MEGA_JOB_TOKEN": "secret-value"},
            "labels": {},
            "volumes": [
                Volume(type="bucket", source="alice/artifacts", mount_path="/output", read_only=False),
                Volume(type="dataset", source="alice/training", mount_path="/input", revision="v1", read_only=True, path="train"),
            ],
            "flavor": "cpu-nano",
            "timeout": None,
            "namespace": None,
            "ssh": False,
        },
    )
    assert "secret-value" not in result.output


def test_jobs_run_loads_dotenv_files_before_repeatable_cli_values(tmp_path, monkeypatch):
    env_file = tmp_path / "job.env"
    env_file.write_text("MODE=file\nREGION=us\n", encoding="utf-8")
    secrets_file = tmp_path / "job.secrets"
    secrets_file.write_text("API_TOKEN=file-secret\n", encoding="utf-8")
    calls = []

    class FakeClient:
        def __init__(self, *, token=None):
            pass

        def run_job(self, **kwargs):
            calls.append(kwargs)
            return _job()

    monkeypatch.setattr(jobs, "MegaHubClient", FakeClient)
    result = CliRunner().invoke(
        jobs.jobs_cli,
        [
            "run", "--detach", "--env-file", str(env_file), "--secrets-file", str(secrets_file),
            "--env", "MODE=cli", "python:3.12-slim", "python", "-c", "print('ok')",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["env"] == {"MODE": "cli", "REGION": "us"}
    assert calls[0]["secrets"] == {"API_TOKEN": "file-secret"}


def test_scheduled_commands_use_hf_names_and_native_client_methods(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def create_scheduled_job(self, **kwargs):
            calls.append(("create", kwargs))
            return _schedule()

        def trigger_scheduled_job(self, schedule_id):
            calls.append(("trigger", schedule_id))
            return _job()

    monkeypatch.setattr(job_schedules, "MegaHubClient", FakeClient)
    runner = CliRunner()
    create = runner.invoke(
        job_schedules.schedules_cli,
        ["run", "@hourly", "python:3.12-slim", "python", "-c", "print('tick')"],
    )
    trigger = runner.invoke(job_schedules.schedules_cli, ["trigger", _schedule().id])

    assert create.exit_code == 0, create.output
    assert trigger.exit_code == 0, trigger.output
    assert calls[1] == (
        "create",
        {
            "image": "python:3.12-slim",
            "command": ["python", "-c", "print('tick')"],
            "schedule": "@hourly",
            "env": {},
            "secrets": {},
            "labels": {},
            "volumes": [],
            "flavor": "cpu-nano",
            "timeout": None,
            "namespace": None,
            "ssh": False,
            "suspend": False,
            "concurrency": False,
        },
    )
    assert calls[-1] == ("trigger", _schedule().id)
    assert runner.invoke(job_schedules.schedules_cli, ["create"]).exit_code != 0
    assert runner.invoke(job_schedules.schedules_cli, ["pause"]).exit_code != 0


def test_jobs_ssh_dry_run_uses_the_returned_private_ingress(monkeypatch):
    class FakeClient:
        def __init__(self, *, token=None):
            pass

        def inspect_job(self, job_id):
            assert job_id == "job-0123456789abcdef0123456789abcdef"
            return SimpleNamespace(
                id=job_id,
                status=SimpleNamespace(ssh_url="ssh://mega-spaces@ssh.mega.space"),
            )

    monkeypatch.setattr(jobs, "MegaHubClient", FakeClient)
    monkeypatch.setattr(jobs.shutil, "which", lambda name: "/usr/bin/cloudflared")
    result = CliRunner().invoke(
        jobs.jobs_cli,
        ["ssh", "job-0123456789abcdef0123456789abcdef", "--dry-run", "python", "-V"],
    )

    assert result.exit_code == 0, result.output
    command = shlex.split(result.output.strip())
    assert command[-2] == "mega-spaces@ssh.mega.space"
    payload = command[-1] + "=" * (-len(command[-1]) % 4)
    assert json.loads(base64.urlsafe_b64decode(payload)) == {
        "kind": "job",
        "job_id": "job-0123456789abcdef0123456789abcdef",
        "command": ["python", "-V"],
    }


def test_jobs_usage_has_stable_json_output(monkeypatch):
    class FakeClient:
        def __init__(self, *, token=None):
            pass

        def get_jobs_usage(self, *, namespace=None):
            return {
                "currency": "USD",
                "namespace": namespace,
                "billedMinutes": 2,
                "amountUSD": 0.000334,
                "jobs": [],
            }

    monkeypatch.setattr(jobs, "MegaHubClient", FakeClient)
    result = CliRunner().invoke(
        jobs.jobs_cli,
        ["usage", "--namespace", "research", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert '"namespace": "research"' in result.output
    assert '"amountUSD": 0.000334' in result.output


def test_jobs_stats_streams_real_metrics_from_the_native_client(monkeypatch):
    class FakeClient:
        def __init__(self, *, token=None):
            pass

        def fetch_job_metrics(self, job_id, *, namespace=None):
            assert job_id == "job-0123456789abcdef0123456789abcdef"
            assert namespace == "research"
            return iter([{"cpu_usage_pct": 12.5, "memory_used_bytes": 1024, "gpus": {}}])

    monkeypatch.setattr(jobs, "MegaHubClient", FakeClient)
    result = CliRunner().invoke(
        jobs.jobs_cli,
        ["stats", "job-0123456789abcdef0123456789abcdef", "--namespace", "research", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert '"cpu_usage_pct": 12.5' in result.output


def test_jobs_uv_commands_use_the_existing_cpu_job_and_scheduler_contracts(monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, *, token=None, library_name=None):
            calls.append(("init", token, library_name))

        def run_uv_job(self, script, **kwargs):
            calls.append(("run", script, kwargs))
            return _job()

        def create_scheduled_uv_job(self, script, **kwargs):
            calls.append(("scheduled", script, kwargs))
            return _schedule()

    monkeypatch.setattr(job_uv, "MegaApi", FakeApi)
    runner = CliRunner()
    immediate = runner.invoke(
        job_uv.uv_cli,
        ["run", "https://example.test/task.py", "--with", "requests", "--ssh", "--detach"],
    )
    scheduled = runner.invoke(
        job_uv.scheduled_uv_cli,
        ["run", "@hourly", "https://example.test/task.py", "--with", "requests", "--ssh", "--suspend"],
    )

    assert immediate.exit_code == 0, immediate.output
    assert scheduled.exit_code == 0, scheduled.output
    assert calls[1] == (
        "run",
        "https://example.test/task.py",
        {
            "script_args": [], "dependencies": ["requests"], "python": None, "image": None,
            "env": {}, "secrets": {}, "flavor": "cpu-nano", "timeout": None,
            "labels": {}, "volumes": [], "ssh": True, "namespace": None,
        },
    )
    assert calls[3] == (
        "scheduled",
        "https://example.test/task.py",
        {
            "script_args": [], "schedule": "@hourly", "suspend": True, "concurrency": False,
            "dependencies": ["requests"], "python": None, "image": None, "env": {},
            "secrets": {}, "flavor": "cpu-nano", "timeout": None, "labels": {},
            "volumes": [], "ssh": True, "namespace": None,
        },
    )


def test_jobs_balance_has_stable_json_output(monkeypatch):
    class FakeClient:
        def __init__(self, *, token=None):
            pass

        def get_compute_billing(self, *, namespace=None):
            return {
                "currency": "USD",
                "owner": {"type": "organization", "id": namespace},
                "balanceMicroUSD": 9_999_666,
                "balanceUSD": 9.999666,
                "spent": {"jobsMicroUSD": 334, "spacesMicroUSD": 0},
                "ledger": [],
            }

    monkeypatch.setattr(jobs, "MegaHubClient", FakeClient)
    result = CliRunner().invoke(
        jobs.jobs_cli,
        ["balance", "--namespace", "research", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert '"balanceUSD": 9.999666' in result.output
    assert '"jobsMicroUSD": 334' in result.output

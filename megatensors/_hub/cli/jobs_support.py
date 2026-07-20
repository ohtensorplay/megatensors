"""Shared parsing and rendering helpers for the native Jobs command groups."""

import os
import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from megatensors.hub import MegaHubError

from megatensors._hub.errors import CLIError
from megatensors._hub._space_api import Volume
from megatensors._hub.utils import parse_mega_mount
from megatensors._hub.utils._dotenv import load_dotenv


F = TypeVar("F", bound=Callable[..., Any])
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def worker_errors_as_cli_errors(command: F) -> F:
    """Translate native Worker failures into the common CLI error surface."""

    @wraps(command)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except MegaHubError as error:
            raise CLIError(str(error)) from error
        except TimeoutError as error:
            raise CLIError(str(error)) from error

    return wrapped  # type: ignore[return-value]


def key_value_entries(values: list[str] | None, name: str) -> dict[str, str]:
    """Parse repeatable KEY=VALUE options without a shell or dotenv parser."""
    result: dict[str, str] = {}
    for raw in values or []:
        key, separator, value = raw.partition("=")
        if not separator or not _KEY.fullmatch(key):
            raise CLIError(f"{name} entries must use KEY=VALUE with a valid environment key.")
        result[key] = value
    return result


def environment_entries(values: list[str] | None, file: Path | None, name: str) -> dict[str, str]:
    """Load optional dotenv entries, then apply repeatable CLI values last."""
    result: dict[str, str] = {}
    if file is not None:
        source = file.expanduser()
        if not source.is_file():
            raise CLIError(f"{name} file does not exist or is not a regular file: {source}")
        try:
            result.update(load_dotenv(source.read_text(encoding="utf-8"), environ=os.environ))
        except OSError as error:
            raise CLIError(f"Could not read {name.lower()} file: {source}") from error
        for key, value in result.items():
            if not _KEY.fullmatch(key) or "\x00" in value:
                raise CLIError(f"{name} file entries must use valid KEY=VALUE lines.")
    result.update(key_value_entries(values, name))
    return result


def job_environment_entries(
    env: list[str] | None,
    env_file: Path | None,
    secret: list[str] | None,
    secrets_file: Path | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build non-secret and secret maps with the same no-overlap contract as Jobs."""
    environment = environment_entries(env, env_file, "Environment")
    secrets = {**environment_entries(None, secrets_file, "Secrets"), **secret_environment_entries(secret)}
    overlap = sorted(set(environment) & set(secrets))
    if overlap:
        raise CLIError(f"Environment and secrets both define '{overlap[0]}'.")
    return environment, secrets


def secret_environment_entries(names: list[str] | None) -> dict[str, str]:
    """Read repeatable secret names from the local process environment."""
    result: dict[str, str] = {}
    for name in names or []:
        if not _KEY.fullmatch(name):
            raise CLIError("Secret names must be valid environment keys and must not include a value.")
        if name not in os.environ:
            raise CLIError(f"Local environment variable '{name}' is not set.")
        result[name] = os.environ[name]
    return result


def job_volume_entries(values: list[str] | None) -> list[Volume]:
    """Parse repeatable canonical MEGA mounts for Jobs and schedules."""
    volumes: list[Volume] = []
    for value in values or []:
        try:
            mount = parse_mega_mount(value)
        except ValueError as error:
            raise CLIError(str(error)) from error
        if mount.source.type not in {"bucket", "model", "dataset", "space"}:
            raise CLIError("Job volumes must reference a bucket, model, dataset, or Space.")
        if mount.source.type != "bucket" and mount.read_only is False:
            raise CLIError("Only Bucket Job volumes can be writable.")
        volumes.append(
            Volume(
                type=mount.source.type,
                source=mount.source.id,
                mount_path=mount.mount_path,
                revision=mount.source.revision,
                read_only=mount.read_only,
                path=mount.source.path_in_repo or None,
            )
        )
    return volumes


def job_record(job: Any) -> dict[str, Any]:
    """Make an upstream-shaped JobInfo safe for CLI tables and JSON output."""
    duration = job.durations.total_secs if job.durations else None
    return {
        "id": job.id,
        "stage": str(job.status.stage),
        "image": job.docker_image,
        "command": list(job.command or []) + list(job.arguments or []),
        "owner": job.owner.name,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "duration_seconds": duration,
        "message": job.status.message,
        "ssh_url": getattr(job.status, "ssh_url", None),
        "labels": job.labels or {},
        "billing": {
            "unit_cost_usd": job.billing.unit_cost_usd,
            "unit_label": job.billing.unit_label,
            "billed_minutes": job.billing.billed_minutes,
            "amount_usd": job.billing.amount_usd,
        } if getattr(job, "billing", None) else None,
        "url": job.url,
    }


def scheduled_job_record(schedule: Any) -> dict[str, Any]:
    """Normalize scheduled Job data for the shared output formatter."""
    last = schedule.status.last_job
    return {
        "id": schedule.id,
        "schedule": schedule.schedule,
        "suspended": bool(schedule.suspend),
        "concurrency": bool(schedule.concurrency),
        "image": schedule.job_spec.docker_image,
        "command": list(schedule.job_spec.command or []) + list(schedule.job_spec.arguments or []),
        "owner": schedule.owner.name,
        "created_at": schedule.created_at.isoformat() if schedule.created_at else None,
        "last_job": last.id if last else None,
        "last_job_at": last.at.isoformat() if last else None,
        "next_run_at": schedule.status.next_job_run_at.isoformat() if schedule.status.next_job_run_at else None,
        "labels": schedule.job_spec.labels or {},
    }

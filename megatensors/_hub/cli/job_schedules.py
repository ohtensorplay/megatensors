"""Recurring Job schedule commands for the native MEGA CLI."""

from pathlib import Path
from typing import Annotated

from megatensors.hub import MegaHubClient

from megatensors._hub.errors import CLIError
from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out
from .job_uv import scheduled_uv_cli
from .jobs_support import job_environment_entries, job_record, job_volume_entries, key_value_entries, scheduled_job_record, worker_errors_as_cli_errors


schedules_cli = typer_factory(help="Create and operate recurring native Jobs.")
schedules_cli.add_group(scheduled_uv_cli, name="uv")


@schedules_cli.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "mega jobs scheduled run @hourly python:3.12-slim python -c 'print(\"hourly\")'",
        "MEGA_JOB_TOKEN=... mega jobs scheduled run '*/15 * * * *' --secret MEGA_JOB_TOKEN python:3.12-slim python task.py",
    ],
)
@worker_errors_as_cli_errors
def schedules_run(
    schedule: Annotated[str, Argument(help="UTC five-field cron expression or @hourly/@daily alias.")],
    image: Annotated[str, Argument(help="Public Docker image for the CPU Job.")],
    command: Annotated[list[str], Argument(help="Program and arguments to execute in the container.")],
    env: Annotated[list[str] | None, Option("-e", "--env", help="Environment entry in KEY=VALUE form. Repeatable.")] = None,
    env_file: Annotated[Path | None, Option("--env-file", help="Read non-secret environment entries from a dotenv file.")] = None,
    secret: Annotated[list[str] | None, Option("--secret", help="Local environment variable name to seal into the Job. Repeatable.")] = None,
    secrets_file: Annotated[Path | None, Option("--secrets-file", help="Read secret environment entries from a dotenv file.")] = None,
    label: Annotated[list[str] | None, Option("-l", "--label", help="Job label in KEY=VALUE form. Repeatable.")] = None,
    volume: Annotated[list[str] | None, Option("-v", "--volume", help="MEGA mount URI. Repeatable.")] = None,
    flavor: Annotated[str, Option("--flavor", help="Live Job flavor; cpu-nano (or compatibility alias cpu-basic) only.")] = "cpu-nano",
    timeout: Annotated[str | None, Option("--timeout", help="Maximum runtime, from 30s through 1h.")] = None,
    namespace: Annotated[str | None, Option("--namespace", help="Personal or organization owner handle.")] = None,
    ssh: Annotated[bool, Option("--ssh", help="Enable private SSH for each running scheduled execution.")] = False,
    suspend: Annotated[bool, Option("--suspend", help="Create the schedule paused.")] = False,
    concurrency: Annotated[bool, Option("--concurrency", help="Allow overlapping scheduled runs.")] = False,
    token: TokenOpt = None,
) -> None:
    """Create a bounded recurring CPU Job with optional Bucket or repository mounts."""
    environment, secrets = job_environment_entries(env, env_file, secret, secrets_file)
    scheduled = MegaHubClient(token=token).create_scheduled_job(
        image=image,
        command=command,
        schedule=schedule,
        env=environment,
        secrets=secrets,
        labels=key_value_entries(label, "Label"),
        volumes=job_volume_entries(volume),
        flavor=flavor,
        timeout=timeout,
        namespace=namespace,
        ssh=ssh,
        suspend=suspend,
        concurrency=concurrency,
    )
    out.dict(scheduled_job_record(scheduled), id_key="id")


@schedules_cli.command("list | ls | ps", examples=["mega jobs scheduled list", "mega jobs scheduled ls --namespace research"])
@worker_errors_as_cli_errors
def schedules_list(
    namespace: Annotated[str | None, Option("--namespace", help="Filter by owner handle.")] = None,
    token: TokenOpt = None,
) -> None:
    """List recurring Jobs owned by the current account."""
    schedules = MegaHubClient(token=token).list_scheduled_jobs(namespace=namespace)
    out.table(
        [
            {
                "id": schedule.id,
                "schedule": schedule.schedule,
                "state": "PAUSED" if schedule.suspend else "ARMED",
                "image": schedule.job_spec.docker_image,
                "next_run_at": schedule.status.next_job_run_at.isoformat() if schedule.status.next_job_run_at else None,
                "last_job": schedule.status.last_job.id if schedule.status.last_job else None,
            }
            for schedule in schedules
        ],
        id_key="id",
    )


@schedules_cli.command("inspect", examples=["mega jobs scheduled inspect <schedule-id>"])
@worker_errors_as_cli_errors
def schedules_inspect(
    schedule_id: Annotated[str, Argument(help="Scheduled Job ID.")],
    token: TokenOpt = None,
) -> None:
    """Show one recurring Job definition and dispatch state."""
    out.dict(scheduled_job_record(MegaHubClient(token=token).inspect_scheduled_job(schedule_id)), id_key="id")


@schedules_cli.command("suspend", examples=["mega jobs scheduled suspend <schedule-id>"])
@worker_errors_as_cli_errors
def schedules_suspend(
    schedule_id: Annotated[str, Argument(help="Scheduled Job ID.")],
    token: TokenOpt = None,
) -> None:
    """Pause future dispatches while preserving the schedule definition."""
    schedule = MegaHubClient(token=token).suspend_scheduled_job(schedule_id)
    out.result("Job schedule paused", schedule_id=schedule.id)


@schedules_cli.command("resume", examples=["mega jobs scheduled resume <schedule-id>"])
@worker_errors_as_cli_errors
def schedules_resume(
    schedule_id: Annotated[str, Argument(help="Scheduled Job ID.")],
    token: TokenOpt = None,
) -> None:
    """Resume future dispatches for one recurring Job."""
    schedule = MegaHubClient(token=token).resume_scheduled_job(schedule_id)
    out.result("Job schedule resumed", schedule_id=schedule.id)


@schedules_cli.command("trigger", examples=["mega jobs scheduled trigger <schedule-id>"])
@worker_errors_as_cli_errors
def schedules_trigger(
    schedule_id: Annotated[str, Argument(help="Scheduled Job ID.")],
    token: TokenOpt = None,
) -> None:
    """Dispatch one immediate run from a saved schedule."""
    out.dict(job_record(MegaHubClient(token=token).trigger_scheduled_job(schedule_id)), id_key="id")


@schedules_cli.command("delete", examples=["mega jobs scheduled delete <schedule-id> --yes"])
@worker_errors_as_cli_errors
def schedules_delete(
    schedule_id: Annotated[str, Argument(help="Scheduled Job ID.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a recurring Job definition without deleting its execution history."""
    out.confirm(f"Delete Job schedule '{schedule_id}'?", yes=yes)
    MegaHubClient(token=token).delete_scheduled_job(schedule_id)
    out.result("Job schedule deleted", schedule_id=schedule_id)

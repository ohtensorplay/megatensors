"""UV Job commands backed by the existing MEGA Job and Bucket contracts."""

from pathlib import Path
from typing import Annotated

from megatensors._hub import MegaApi
from megatensors.hub import MegaHubClient

from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out
from .jobs_support import (
    job_environment_entries,
    job_record,
    job_volume_entries,
    key_value_entries,
    scheduled_job_record,
    worker_errors_as_cli_errors,
)


uv_cli = typer_factory(help="Run UV scripts through the current MEGA CPU Job runner.")
scheduled_uv_cli = typer_factory(help="Schedule UV scripts through the current MEGA CPU Job runner.")


@uv_cli.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "mega jobs uv run https://example.test/task.py --with requests",
        "mega jobs uv run --detach task.py -- --input /data/input.jsonl",
    ],
)
@worker_errors_as_cli_errors
def jobs_uv_run(
    script: Annotated[str, Argument(help="Local script, HTTPS URL, or UV command.")],
    script_args: Annotated[list[str] | None, Argument(help="Arguments passed to the script or command.")] = None,
    dependency: Annotated[list[str] | None, Option("--with", help="Dependency passed to `uv run --with`. Repeatable.")] = None,
    python: Annotated[str | None, Option("--python", help="Python version or interpreter for UV.")] = None,
    image: Annotated[str | None, Option("--image", help="Public image that contains uv.")] = None,
    env: Annotated[list[str] | None, Option("-e", "--env", help="Environment entry in KEY=VALUE form. Repeatable.")] = None,
    env_file: Annotated[Path | None, Option("--env-file", help="Read non-secret environment entries from a dotenv file.")] = None,
    secret: Annotated[list[str] | None, Option("--secret", help="Local environment variable name to seal into the Job. Repeatable.")] = None,
    secrets_file: Annotated[Path | None, Option("--secrets-file", help="Read secret environment entries from a dotenv file.")] = None,
    label: Annotated[list[str] | None, Option("-l", "--label", help="Job label in KEY=VALUE form. Repeatable.")] = None,
    volume: Annotated[list[str] | None, Option("-v", "--volume", help="MEGA mount URI. Repeatable.")] = None,
    flavor: Annotated[str, Option("--flavor", help="Live Job flavor; cpu-nano (or compatibility alias cpu-basic) only.")] = "cpu-nano",
    timeout: Annotated[str | None, Option("--timeout", help="Maximum runtime, from 30s through 1h.")] = None,
    namespace: Annotated[str | None, Option("--namespace", help="Personal or organization owner handle.")] = None,
    ssh: Annotated[bool, Option("--ssh", help="Enable private SSH while the Job is running.")] = False,
    detach: Annotated[bool, Option("-d", "--detach", help="Return after the runner accepts the Job.")] = False,
    token: TokenOpt = None,
) -> None:
    """Run a UV script or command on the native CPU Job runner."""
    environment, secrets = job_environment_entries(env, env_file, secret, secrets_file)
    job = MegaApi(token=token, library_name="mega-cli").run_uv_job(
        script,
        script_args=list(script_args or []),
        dependencies=list(dependency or []),
        python=python,
        image=image,
        env=environment,
        secrets=secrets,
        flavor=flavor,
        timeout=timeout,
        labels=key_value_entries(label, "Label"),
        volumes=job_volume_entries(volume),
        ssh=ssh,
        namespace=namespace,
    )
    out.result("UV Job accepted", job_id=job.id, stage=job.status.stage, ssh=ssh, url=job.url)
    if detach:
        out.hint(f"Use `mega jobs logs --follow {job.id}` or `mega jobs wait {job.id}` to observe this run.")
        return
    client = MegaHubClient(token=token)
    final = client.wait_for_job(job.id, timeout=5_400)
    for line in client.fetch_job_logs(final.id, tail=500):
        out.text(line)
    out.dict(job_record(final), id_key="id")


@scheduled_uv_cli.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "mega jobs scheduled uv run @hourly https://example.test/task.py --with requests",
        "mega jobs scheduled uv run '*/15 * * * *' task.py --suspend",
    ],
)
@worker_errors_as_cli_errors
def jobs_scheduled_uv_run(
    schedule: Annotated[str, Argument(help="UTC five-field cron expression or @hourly/@daily alias.")],
    script: Annotated[str, Argument(help="Local script, HTTPS URL, or UV command.")],
    script_args: Annotated[list[str] | None, Argument(help="Arguments passed to the script or command.")] = None,
    dependency: Annotated[list[str] | None, Option("--with", help="Dependency passed to `uv run --with`. Repeatable.")] = None,
    python: Annotated[str | None, Option("--python", help="Python version or interpreter for UV.")] = None,
    image: Annotated[str | None, Option("--image", help="Public image that contains uv.")] = None,
    env: Annotated[list[str] | None, Option("-e", "--env", help="Environment entry in KEY=VALUE form. Repeatable.")] = None,
    env_file: Annotated[Path | None, Option("--env-file", help="Read non-secret environment entries from a dotenv file.")] = None,
    secret: Annotated[list[str] | None, Option("--secret", help="Local environment variable name to seal into the Job. Repeatable.")] = None,
    secrets_file: Annotated[Path | None, Option("--secrets-file", help="Read secret environment entries from a dotenv file.")] = None,
    label: Annotated[list[str] | None, Option("-l", "--label", help="Job label in KEY=VALUE form. Repeatable.")] = None,
    volume: Annotated[list[str] | None, Option("-v", "--volume", help="MEGA mount URI. Repeatable.")] = None,
    flavor: Annotated[str, Option("--flavor", help="Live Job flavor; cpu-nano (or compatibility alias cpu-basic) only.")] = "cpu-nano",
    timeout: Annotated[str | None, Option("--timeout", help="Maximum runtime, from 30s through 1h.")] = None,
    namespace: Annotated[str | None, Option("--namespace", help="Personal or organization owner handle.")] = None,
    ssh: Annotated[bool, Option("--ssh", help="Enable private SSH for each running execution.")] = False,
    suspend: Annotated[bool, Option("--suspend", help="Create the schedule paused.")] = False,
    concurrency: Annotated[bool, Option("--concurrency", help="Allow overlapping scheduled runs.")] = False,
    token: TokenOpt = None,
) -> None:
    """Create a recurring UV Job through the native scheduler."""
    environment, secrets = job_environment_entries(env, env_file, secret, secrets_file)
    scheduled = MegaApi(token=token, library_name="mega-cli").create_scheduled_uv_job(
        script,
        script_args=list(script_args or []),
        schedule=schedule,
        suspend=suspend,
        concurrency=concurrency,
        dependencies=list(dependency or []),
        python=python,
        image=image,
        env=environment,
        secrets=secrets,
        flavor=flavor,
        timeout=timeout,
        labels=key_value_entries(label, "Label"),
        volumes=job_volume_entries(volume),
        ssh=ssh,
        namespace=namespace,
    )
    out.dict(scheduled_job_record(scheduled), id_key="id")

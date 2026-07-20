"""Bounded native MEGA Jobs CLI commands."""

import base64
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import click

from megatensors.hub import MegaHubClient

from megatensors._hub.errors import CLIError
from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out
from .job_schedules import schedules_cli
from .job_uv import uv_cli
from .jobs_support import job_environment_entries, job_record, job_volume_entries, key_value_entries, worker_errors_as_cli_errors


jobs_cli = typer_factory(help="Run bounded CPU Jobs on the protected MEGA VPS runner.")
jobs_cli.add_group(schedules_cli, name="scheduled")
jobs_cli.add_group(uv_cli, name="uv")


@jobs_cli.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    examples=[
        "mega jobs run python:3.12-slim python -c 'print(\"hello\")'",
        "mega jobs run --detach -e MODE=check python:3.12-slim python task.py",
        "mega jobs run -v mega://buckets/owner/artifacts:/output:rw python:3.12-slim python task.py",
        "mega jobs run --ssh --detach python:3.12-slim sleep 3600",
        "MEGA_JOB_TOKEN=... mega jobs run --secret MEGA_JOB_TOKEN python:3.12-slim python task.py",
    ],
)
@worker_errors_as_cli_errors
def jobs_run(
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
    ssh: Annotated[bool, Option("--ssh", help="Enable private SSH while the Job is running.")] = False,
    detach: Annotated[bool, Option("-d", "--detach", help="Return after the runner accepts the Job.")] = False,
    token: TokenOpt = None,
) -> None:
    """Run one CPU Nano container with optional Bucket or repository mounts."""
    client = MegaHubClient(token=token)
    environment, secrets = job_environment_entries(env, env_file, secret, secrets_file)
    job = client.run_job(
        image=image,
        command=command,
        env=environment,
        secrets=secrets,
        labels=key_value_entries(label, "Label"),
        volumes=job_volume_entries(volume),
        flavor=flavor,
        timeout=timeout,
        namespace=namespace,
        ssh=ssh,
    )
    out.result("Job accepted", job_id=job.id, stage=job.status.stage, ssh=ssh, url=job.url)
    if detach:
        hint = f"Use `mega jobs logs --follow {job.id}` or `mega jobs wait {job.id}` to observe this run."
        if ssh:
            hint += f" Once it is RUNNING, connect with `mega jobs ssh {job.id}`."
        out.hint(hint)
        return
    final = client.wait_for_job(job.id, timeout=5_400)
    for line in client.fetch_job_logs(final.id, tail=500):
        out.text(line)
    if str(final.status.stage) != "COMPLETED":
        detail = f": {final.status.message}" if final.status.message else ""
        raise CLIError(f"Job {final.id} finished with {final.status.stage}{detail}")
    out.result("Job completed", job_id=final.id)


@jobs_cli.command("list | ls | ps", examples=["mega jobs list", "mega jobs ps --status RUNNING", "mega jobs ls --label lane=release"])
@worker_errors_as_cli_errors
def jobs_list(
    limit: Annotated[int, Option("--limit", help="Maximum Jobs to return (1-100).", min=1)] = 30,
    status: Annotated[list[str] | None, Option("--status", help="Filter by stage. Repeatable.")] = None,
    label: Annotated[list[str] | None, Option("-l", "--label", help="Filter by label KEY=VALUE. Repeatable.")] = None,
    namespace: Annotated[str | None, Option("--namespace", help="Filter by owner handle.")] = None,
    token: TokenOpt = None,
) -> None:
    """List Job execution history for the authenticated account."""
    jobs = MegaHubClient(token=token).list_jobs(
        limit=limit,
        stages=status,
        labels=key_value_entries(label, "Label"),
        namespace=namespace,
    )
    out.table(
        [
            {
                "id": job.id,
                "stage": job.status.stage,
                "image": job.docker_image,
                "owner": job.owner.name,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "message": job.status.message,
            }
            for job in jobs
        ],
        id_key="id",
    )


@jobs_cli.command("inspect", examples=["mega jobs inspect <job-id>"])
@worker_errors_as_cli_errors
def jobs_inspect(
    job_id: Annotated[str, Argument(help="Job ID.")],
    token: TokenOpt = None,
) -> None:
    """Show current state, timing, labels, and retained metadata for one Job."""
    out.dict(job_record(MegaHubClient(token=token).inspect_job(job_id)), id_key="id")


@jobs_cli.command(
    "ssh",
    examples=["mega jobs ssh <job-id>", "mega jobs ssh <job-id> python -c 'print(1)'", "mega jobs ssh <job-id> --dry-run"],
    context_settings={"ignore_unknown_options": True},
)
@worker_errors_as_cli_errors
def jobs_ssh(
    job_id: Annotated[str, Argument(help="Running Job ID created with --ssh or ssh=True.")],
    remote_command: Annotated[list[str] | None, Argument(help="Optional command to run instead of an interactive shell.")] = None,
    identity_file: Annotated[Path | None, Option("-i", "--identity-file", help="SSH private key.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", help="Print the SSH command without running it.")] = False,
    token: TokenOpt = None,
) -> None:
    """SSH into a running Job through the shared Access-protected ingress."""
    remote_args = list(remote_command or [])
    while remote_args:
        if remote_args[0] == "--dry-run":
            dry_run = True
            remote_args.pop(0)
            continue
        if remote_args[0] in {"-i", "--identity-file"} and len(remote_args) >= 2:
            identity_file = Path(remote_args[1])
            del remote_args[:2]
            continue
        break
    job = MegaHubClient(token=token).inspect_job(job_id)
    ssh_url = job.status.ssh_url
    if not ssh_url:
        raise CLIError("Job SSH is available only while a Job created with --ssh is RUNNING.")
    endpoint = urlsplit(ssh_url)
    host = os.environ.get("MEGA_JOB_SSH_HOST") or endpoint.hostname
    if endpoint.scheme != "ssh" or not host or not all(part and part.replace("-", "").isalnum() for part in host.split(".")):
        raise CLIError("Job returned an invalid SSH endpoint; MEGA_JOB_SSH_HOST must be a valid hostname.")
    request = base64.urlsafe_b64encode(json.dumps({
        "kind": "job",
        "job_id": job.id,
        "command": remote_args,
    }, separators=(",", ":")).encode()).decode().rstrip("=")
    cloudflared = shutil.which("cloudflared") or "cloudflared"
    command = [
        "ssh",
        "-o", f"ProxyCommand={cloudflared} access ssh --hostname %h",
        "-o", "RequestTTY=force",
    ]
    if identity_file is not None:
        command.extend(["-i", str(identity_file.expanduser())])
    command.extend([f"{endpoint.username or 'mega-spaces'}@{host}", request])
    if dry_run:
        click.echo(shlex.join(command))
        return
    raise SystemExit(subprocess.call(command))


@jobs_cli.command("logs", examples=["mega jobs logs <job-id>", "mega jobs logs --follow --tail 100 <job-id>"])
@worker_errors_as_cli_errors
def jobs_logs(
    job_id: Annotated[str, Argument(help="Job ID.")],
    follow: Annotated[bool, Option("-f", "--follow", help="Stream new output until the container exits.")] = False,
    tail: Annotated[int, Option("-n", "--tail", help="Retained log lines to request (1-5000).", min=1)] = 200,
    token: TokenOpt = None,
) -> None:
    """Print retained Job output; logs remain available for the configured retention period."""
    for line in MegaHubClient(token=token).fetch_job_logs(job_id, follow=follow, tail=tail):
        out.text(line)


@jobs_cli.command("wait", examples=["mega jobs wait <job-id>", "mega jobs wait <job-id> --timeout 900"])
@worker_errors_as_cli_errors
def jobs_wait(
    job_id: Annotated[str, Argument(help="Job ID.")],
    timeout: Annotated[int | None, Option("--timeout", help="Maximum seconds to wait.", min=1)] = None,
    interval: Annotated[float, Option("--interval", help="Polling interval in seconds.")] = 2.0,
    token: TokenOpt = None,
) -> None:
    """Wait until a Job completes, cancels, or errors."""
    job = MegaHubClient(token=token).wait_for_job(job_id, timeout=timeout, poll_interval=interval)
    out.dict(job_record(job), id_key="id")
    if str(job.status.stage) != "COMPLETED":
        raise CLIError(f"Job {job.id} finished with {job.status.stage}")


@jobs_cli.command("cancel", examples=["mega jobs cancel <job-id>", "mega jobs cancel <job-id> --yes"])
@worker_errors_as_cli_errors
def jobs_cancel(
    job_id: Annotated[str, Argument(help="Job ID.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    token: TokenOpt = None,
) -> None:
    """Cancel a scheduling or running Job."""
    out.confirm(f"Cancel Job '{job_id}'?", yes=yes)
    job = MegaHubClient(token=token).cancel_job(job_id)
    out.result("Job canceled", job_id=job.id, stage=job.status.stage)


@jobs_cli.command("hardware", examples=["mega jobs hardware"])
@worker_errors_as_cli_errors
def jobs_hardware(token: TokenOpt = None) -> None:
    """List hardware profiles currently enabled by the native runner."""
    out.table(
        [
            {
                "name": item.name,
                "description": item.pretty_name,
                "cpu": item.cpu,
                "ram": item.ram,
                "storage": item.ephemeral_storage,
                "cost": item.unit_cost_usd,
                "unit": item.unit_label,
            }
            for item in MegaHubClient(token=token).list_jobs_hardware()
        ],
        id_key="name",
    )


@jobs_cli.command("stats", examples=["mega jobs stats <job-id>", "mega jobs stats <job-id> --namespace research"])
@worker_errors_as_cli_errors
def jobs_stats(
    job_id: Annotated[str, Argument(help="Running Job ID.")],
    namespace: Annotated[str | None, Option("--namespace", help="Personal or organization owner handle.")] = None,
    token: TokenOpt = None,
) -> None:
    """Stream live CPU, memory, and network metrics for a running Job."""
    for metric in MegaHubClient(token=token).fetch_job_metrics(job_id, namespace=namespace):
        out.dict(metric)


@jobs_cli.command("usage", examples=["mega jobs usage", "mega jobs usage --namespace research"])
@worker_errors_as_cli_errors
def jobs_usage(
    namespace: Annotated[str | None, Option("--namespace", help="Personal or organization billing namespace.")] = None,
    token: TokenOpt = None,
) -> None:
    """Show accrued per-minute Job usage and USD cost."""
    out.dict(MegaHubClient(token=token).get_jobs_usage(namespace=namespace))


@jobs_cli.command("balance", examples=["mega jobs balance", "mega jobs balance --namespace research"])
@worker_errors_as_cli_errors
def jobs_balance(
    namespace: Annotated[str | None, Option("--namespace", help="Personal or organization billing namespace.")] = None,
    token: TokenOpt = None,
) -> None:
    """Show prepaid compute balance, Job and Space spend, and recent ledger entries."""
    out.dict(MegaHubClient(token=token).get_compute_billing(namespace=namespace))

"""Commands to create and manage MEGA cloud sandboxes."""

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Iterator

import click

from megatensors.hub import MegaHubClient
from megatensors._hub._sandbox import (
    DEFAULT_IDLE_TIMEOUT,
    MODE_LABEL,
    MODE_POOL,
    POOL_LABEL,
    SANDBOX_LABEL,
    SHARED_ID_SEP,
    Sandbox as JobsSandbox,
    _split_sandbox_id,
)
from megatensors._hub._native_sandbox import (
    NativeSandbox as Sandbox,
    NativeSandboxPool as SandboxPool,
    NativeSandboxProcess as SandboxProcess,
)
from megatensors._hub._space_api import Volume
from megatensors._hub.errors import CLIError, SandboxError
from megatensors._hub.mega_api import MegaApi
from megatensors._hub.utils import parse_mega_mount
from megatensors._hub.utils._dotenv import load_dotenv

from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out


sandbox_cli = typer_factory(help="Run and manage isolated MEGA Sandbox sessions.")
pool_cli = typer_factory(help="Manage native warm-template Sandbox pools.")
process_cli = typer_factory(help="List and stop background processes in a sandbox.")
sandbox_cli.add_group(pool_cli, name="pool")
sandbox_cli.add_group(process_cli, name="process")

SandboxIdArg = Annotated[str, Argument(help="Sandbox id printed by `mega sandbox create`.")]
FlavorOpt = Annotated[str | None, Option("--flavor", help="Sandbox hardware flavor.")]
NamespaceOpt = Annotated[str | None, Option("--namespace", help="Personal or organization namespace.")]
NATIVE_SANDBOXES_PER_BROKER = 4


def _api(token: str | None) -> MegaApi:
    client = MegaHubClient(token=token)
    return MegaApi(endpoint=client.endpoint, token=client.token)


def _parse_env(values: list[str] | None, file: Path | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values or []:
        key, separator, value = raw.partition("=")
        if not key:
            raise CLIError("Environment keys cannot be empty.")
        if separator:
            result[key] = value
        elif key in os.environ:
            result[key] = os.environ[key]
        else:
            raise CLIError(f"Local environment variable '{key}' is not set.")
    if file is not None:
        if not file.is_file():
            raise CLIError(f"Environment file '{file}' does not exist.")
        result.update(load_dotenv(file.read_text(), environ=os.environ))
    return result


def _parse_volumes(values: list[str] | None) -> list[Volume]:
    volumes: list[Volume] = []
    for value in values or []:
        try:
            mount = parse_mega_mount(value)
        except ValueError as error:
            raise CLIError(str(error)) from error
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


@contextmanager
def _connect(sandbox_id: str, *, namespace: str | None, token: str | None) -> Iterator[Sandbox]:
    sandbox = (
        Sandbox.connect(sandbox_id, namespace=namespace, token=token)
        if sandbox_id.startswith("sandbox-")
        else JobsSandbox.connect(sandbox_id, namespace=namespace, token=token)
    )
    try:
        yield sandbox
    finally:
        sandbox.close()


@sandbox_cli.command("create", examples=["mega sandbox create", "mega sandbox create python:3.13"])
def sandbox_create(
    image: Annotated[str | None, Argument(help="Runtime image; currently python:3.13 only.")] = None,
    pool: Annotated[str | None, Option("--pool", help="Spawn from an existing warm pool.")] = None,
    flavor: FlavorOpt = None,
    idle_timeout: Annotated[str | None, Option("--idle-timeout", help="Idle shutdown duration, for example 10m.")] = None,
    env: Annotated[list[str] | None, Option("-e", "--env", help="KEY=VALUE or local key. Repeatable.")] = None,
    secrets: Annotated[list[str] | None, Option("-s", "--secrets", help="KEY=VALUE or local key. Repeatable.")] = None,
    env_file: Annotated[Path | None, Option("--env-file", help="Read variables from a dotenv file.")] = None,
    secrets_file: Annotated[Path | None, Option("--secrets-file", help="Read secrets from a dotenv file.")] = None,
    volume: Annotated[list[str] | None, Option("-v", "--volume", help="MEGA mount URI. Repeatable.")] = None,
    namespace: NamespaceOpt = None,
    forward_mega_token: Annotated[bool, Option("--forward-mega-token", help="Inject MEGA_TOKEN into the sandbox.")] = False,
    allow_egress: Annotated[bool, Option("--allow-egress", help="Allow outbound network access.")] = False,
    max_lifetime: Annotated[str, Option("--max-lifetime", help="Maximum Sandbox lifetime.")] = "1h",
    token: TokenOpt = None,
) -> None:
    """Create a dedicated sandbox, or fork one from a native warm template."""
    start = time.time()
    idle: str | int = idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT
    if pool is not None:
        if image is not None or flavor is not None or volume:
            raise CLIError("--pool fixes image and flavor, and pooled sandboxes do not accept volumes.")
        sandbox = SandboxPool.connect(pool, namespace=namespace, token=token).create(
            env=_parse_env(env, env_file), secrets=_parse_env(secrets, secrets_file),
            idle_timeout=idle, max_lifetime=max_lifetime, forward_mega_token=forward_mega_token,
        )
        out.result("Sandbox ready", id=sandbox.id, host=sandbox.host_id, pool=pool, elapsed=f"{time.time() - start:.1f}s")
        return
    if image not in (None, "python:3.13"):
        raise CLIError("The native Sandbox service currently supports only the managed python:3.13 runtime.")
    if flavor not in (None, "cpu-basic"):
        raise CLIError("The native Sandbox service currently supports only the cpu-basic flavor.")
    volumes = _parse_volumes(volume)
    environment = _parse_env(env, env_file)
    if forward_mega_token:
        resolved_token = _api(token).token
        if not resolved_token:
            raise CLIError("--forward-mega-token requires an authenticated MEGA token.")
        environment["MEGA_TOKEN"] = resolved_token
    sandbox = Sandbox.create(
        runtime="python-3.13",
        flavor=flavor or "cpu-basic",
        idle_timeout=idle,
        max_lifetime=max_lifetime,
        env=environment,
        secrets=_parse_env(secrets, secrets_file),
        namespace=namespace,
        allow_egress=allow_egress,
        volumes=volumes,
        token=token,
    )
    sandbox.close()
    out.result("Sandbox ready", id=sandbox.id, image=sandbox.image, elapsed=f"{time.time() - start:.1f}s")


@sandbox_cli.command("exec", context_settings={"ignore_unknown_options": True}, examples=["mega sandbox exec <id> -- python -V"])
def sandbox_exec(
    sandbox_id: SandboxIdArg,
    command: Annotated[list[str], Argument(help="Command to run.")],
    workdir: Annotated[str | None, Option("-w", "--workdir", help="Working directory.")] = None,
    env: Annotated[list[str] | None, Option("-e", "--env", help="KEY=VALUE. Repeatable.")] = None,
    env_file: Annotated[Path | None, Option("--env-file", help="Read variables from a dotenv file.")] = None,
    exec_timeout: Annotated[float | None, Option("--timeout", help="Kill the command after this many seconds.")] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Run a command in a sandbox and stream its output."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        result = sandbox.run(
            list(command),
            env=_parse_env(env, env_file),
            cwd=workdir,
            timeout=exec_timeout,
            on_stdout=lambda value: (sys.stdout.write(value), sys.stdout.flush()),
            on_stderr=lambda value: (sys.stderr.write(value), sys.stderr.flush()),
            check=False,
        )
    if result.timed_out:
        raise click.exceptions.Exit(code=result.exit_code or 124)
    if result.exit_code != 0:
        raise click.exceptions.Exit(code=result.exit_code if result.exit_code is not None else 1)


@sandbox_cli.command("spawn", context_settings={"ignore_unknown_options": True}, examples=["mega sandbox spawn <id> -- python -m http.server"])
def sandbox_spawn(
    sandbox_id: SandboxIdArg,
    command: Annotated[list[str], Argument(help="Background command to run.")],
    workdir: Annotated[str | None, Option("-w", "--workdir", help="Working directory.")] = None,
    env: Annotated[list[str] | None, Option("-e", "--env", help="KEY=VALUE. Repeatable.")] = None,
    env_file: Annotated[Path | None, Option("--env-file", help="Read variables from a dotenv file.")] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Start a background process and return its pid."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        process = sandbox.run(list(command), env=_parse_env(env, env_file), cwd=workdir, background=True)
    out.result("Process started", sandbox=sandbox_id, pid=process.pid)


@sandbox_cli.command("cp", examples=["mega sandbox cp data.csv <id>:/data/data.csv"])
def sandbox_cp(
    src: Annotated[str, Argument(help="Local path or <sandbox_id>:<path>.")],
    dst: Annotated[str, Argument(help="Local path or <sandbox_id>:<path>.")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Copy a file between the local machine and a sandbox."""
    def parse(value: str) -> tuple[str | None, str]:
        if ":" in value and not value.startswith((".", "/", "~")):
            sandbox_id, path = value.split(":", 1)
            if len(sandbox_id) > 1:
                return sandbox_id, path
        return None, value

    src_sandbox, src_path = parse(src)
    dst_sandbox, dst_path = parse(dst)
    if (src_sandbox is None) == (dst_sandbox is None):
        raise CLIError("Exactly one of SRC and DST must be a sandbox path.")
    if src_sandbox is not None:
        with _connect(src_sandbox, namespace=namespace, token=token) as sandbox:
            sandbox.files.download(src_path, dst_path)
    else:
        assert dst_sandbox is not None
        with _connect(dst_sandbox, namespace=namespace, token=token) as sandbox:
            sandbox.files.upload(src_path, dst_path)
    out.result("Copied", src=src, dst=dst)


@sandbox_cli.command("kill", examples=["mega sandbox kill <id>", "mega sandbox kill --all"])
def sandbox_kill(
    sandbox_id: Annotated[str | None, Argument(help="Sandbox or shared host id.")] = None,
    all_: Annotated[bool, Option("--all", help="Terminate every sandbox in the namespace.")] = False,
    yes: Annotated[bool, Option("-y", "--yes", help="Skip confirmation.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Terminate a sandbox, host, or all sandbox Jobs."""
    api = _api(token)
    if all_:
        sandboxes = [item for item in Sandbox.list(token=token, namespace=namespace) if item.get("status") in {"CREATING", "RUNNING", "STOPPING"}]
        if not sandboxes:
            out.text("No running sandboxes.")
            return
        out.confirm(f"Terminate {len(sandboxes)} sandbox(s)?", yes=yes)
        for item in sandboxes:
            Sandbox.connect(str(item["id"]), token=token, namespace=namespace).kill()
        out.result("Terminated", sandboxes=len(sandboxes))
        return
    if sandbox_id is None:
        raise CLIError("Provide a sandbox id, host id, or --all.")
    if sandbox_id.startswith("sandbox-"):
        Sandbox.connect(sandbox_id, token=token, namespace=namespace).kill()
        out.result("Sandbox terminated", id=sandbox_id)
        return
    sid, ns = _split_sandbox_id(sandbox_id, namespace)
    if SHARED_ID_SEP not in sid:
        job = api.inspect_job(job_id=sid, namespace=ns)
        if (job.labels or {}).get(MODE_LABEL) == MODE_POOL:
            out.confirm(f"Terminate shared host {sid} and all its sandboxes?", yes=yes)
            api.cancel_job(job_id=job.id, namespace=job.owner.name)
            out.result("Host terminated", id=sid)
            return
    try:
        with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
            sandbox.kill()
    except SandboxError as error:
        raise CLIError(str(error)) from error
    out.result("Sandbox terminated", id=sandbox_id)


@pool_cli.command("create", examples=["mega sandbox pool create", "mega sandbox pool create python:3.13"])
def pool_create(
    image: Annotated[str | None, Argument(help="Docker image for pool hosts.")] = None,
    flavor: FlavorOpt = None,
    per_host: Annotated[int, Option("--per-host", min=1, help="Broker capacity; currently fixed at 4.")] = NATIVE_SANDBOXES_PER_BROKER,
    max_hosts: Annotated[int | None, Option("--max-hosts", min=1, help="Broker count; currently fixed at 1.")] = None,
    idle_timeout: Annotated[str | None, Option("--idle-timeout", help="Warm-template idle refresh duration (maximum 1h).")] = None,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Create and warm a sandbox pool."""
    if per_host != NATIVE_SANDBOXES_PER_BROKER:
        raise CLIError("--per-host is fixed at 4 for the native Cloud Run Sandbox broker.")
    if max_hosts not in (None, 1):
        raise CLIError("--max-hosts is fixed at 1 until native pools support multi-broker scaling.")
    start = time.time()
    pool = SandboxPool.create_pool(
        image=image or "python:3.13",
        flavor=flavor or "cpu-basic",
        per_host=per_host,
        max_hosts=max_hosts,
        idle_timeout=idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT,
        namespace=namespace,
        token=token,
    )
    out.result("Pool created", id=pool.name, image=pool.image, flavor=pool.flavor, host=pool.host_ids[0], elapsed=f"{time.time() - start:.1f}s")


@pool_cli.command("ls | list", examples=["mega sandbox pool ls"])
def pool_list(namespace: NamespaceOpt = None, token: TokenOpt = None) -> None:
    """List running sandbox pools."""
    pools = SandboxPool.list(namespace=namespace, token=token)
    out.table([
        {"id": pool["id"], "image": pool.get("image"), "flavor": pool.get("flavor"),
         "per_host": pool.get("perHost"), "hosts": pool.get("maxHosts"), "stage": pool.get("stage")}
        for pool in pools
    ], id_key="id")


@pool_cli.command("delete | rm", examples=["mega sandbox pool delete <pool_id>"])
def pool_delete(
    pool_id: Annotated[str, Argument(help="Pool id.")],
    yes: Annotated[bool, Option("-y", "--yes", help="Skip confirmation.")] = False,
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Delete a native warm-template Sandbox pool."""
    pool = SandboxPool.connect(pool_id, namespace=namespace, token=token)
    out.confirm(f"Delete warm Sandbox pool '{pool_id}'?", yes=yes)
    pool.delete()
    out.result("Pool deleted", id=pool_id, hosts_terminated=1)


def _process_status(process: SandboxProcess) -> str:
    return "running" if process.running else ("exited" if process.exit_code is None else f"exited ({process.exit_code})")


@process_cli.command("ls | list", examples=["mega sandbox process ls <id>"])
def process_list(sandbox_id: SandboxIdArg, namespace: NamespaceOpt = None, token: TokenOpt = None) -> None:
    """List background processes in a sandbox."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        processes = sandbox.processes()
    out.table([{"pid": process.pid, "status": _process_status(process), "cmd": process.cmd if isinstance(process.cmd, str) else " ".join(process.cmd)} for process in processes], id_key="pid")


@process_cli.command("kill", examples=["mega sandbox process kill <id> <pid>"])
def process_kill(
    sandbox_id: SandboxIdArg,
    pid: Annotated[int, Argument(help="Process id.")],
    namespace: NamespaceOpt = None,
    token: TokenOpt = None,
) -> None:
    """Stop a background process in a sandbox."""
    with _connect(sandbox_id, namespace=namespace, token=token) as sandbox:
        process = next((item for item in sandbox.processes() if item.pid == pid), None)
        if process is None:
            raise CLIError(f"No process with pid {pid} in sandbox {sandbox_id}.")
        process.kill()
    out.result("Process stopped", sandbox=sandbox_id, pid=pid)

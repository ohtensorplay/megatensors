"""Client for MEGA's authenticated first-class Sandbox Session API."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from megatensors.hub import MegaHubClient

from ._sandbox import SandboxCommandResult
from .errors import SandboxError
from .mega_api import MegaApi
from .utils import get_session, mega_raise_for_status


@dataclass
class NativeSandboxProcess:
    pid: int
    cmd: str | list[str]
    _sandbox: "NativeSandbox"
    running: bool = True
    exit_code: int | None = None

    def kill(self) -> None:
        result = self._sandbox.run(
            ["python3", "-c", "import os,signal,sys; os.killpg(int(sys.argv[1]), signal.SIGTERM)", str(self.pid)],
            check=False,
        )
        if result.exit_code not in (0, 1):
            raise SandboxError(result.stderr or f"Could not stop process {self.pid}.")


class NativeSandboxFiles:
    _CHUNK_BYTES = 96 * 1024

    def __init__(self, sandbox: "NativeSandbox") -> None:
        self._sandbox = sandbox

    def upload(self, local_path: str | Path, path: str) -> None:
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        self._sandbox.run(["python3", "-c", "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'')", path])
        with source.open("rb") as handle:
            while chunk := handle.read(self._CHUNK_BYTES):
                encoded = base64.b64encode(chunk).decode("ascii")
                self._sandbox.run(
                    ["python3", "-c", "import base64,sys; open(sys.argv[1],'ab').write(base64.b64decode(sys.stdin.read()))", path],
                    stdin=encoded,
                )

    def download(self, path: str, local_path: str | Path) -> None:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stat = self._sandbox.run(["python3", "-c", "import os,sys; print(os.path.getsize(sys.argv[1]))", path])
        try:
            size = int(stat.stdout.strip())
        except ValueError as error:
            raise SandboxError(f"Could not stat remote file '{path}'.") from error
        with target.open("wb") as handle:
            for offset in range(0, size, self._CHUNK_BYTES):
                result = self._sandbox.run(
                    ["python3", "-c", "import base64,sys; f=open(sys.argv[1],'rb'); f.seek(int(sys.argv[2])); print(base64.b64encode(f.read(int(sys.argv[3]))).decode())", path, str(offset), str(self._CHUNK_BYTES)]
                )
                handle.write(base64.b64decode(result.stdout.strip()))


class NativeSandbox:
    """One Session allocated through ``/api/sandboxes``."""

    def __init__(self, data: dict[str, Any], *, endpoint: str, token: str | None) -> None:
        self.raw = data
        self.id = str(data["id"])
        self.image = str(data.get("runtime") or "python-3.13")
        self.host_id = None
        self._endpoint = endpoint.rstrip("/")
        self._api = MegaApi(endpoint=endpoint, token=token)
        self._headers = self._api._build_mega_headers(token=token)
        self.files = NativeSandboxFiles(self)

    @classmethod
    def create(
        cls,
        *,
        runtime: str = "python-3.13",
        flavor: str = "cpu-basic",
        idle_timeout: str | int = "10m",
        max_lifetime: str | int = "1h",
        env: dict[str, str] | None = None,
        secrets: dict[str, str] | None = None,
        namespace: str | None = None,
        allow_egress: bool = False,
        volumes: list[Any] | None = None,
        pool_id: str | None = None,
        token: str | None = None,
    ) -> "NativeSandbox":
        client = MegaHubClient(token=token)
        api = MegaApi(endpoint=client.endpoint, token=client.token)
        body: dict[str, Any] = {
            "runtime": runtime,
            "flavor": flavor,
            "idleTimeoutSeconds": idle_timeout,
            "maxLifetimeSeconds": max_lifetime,
            "environment": env or {},
            "secrets": secrets or {},
            "allowEgress": allow_egress,
            "volumes": [_volume_payload(volume) for volume in (volumes or [])],
        }
        if pool_id is not None:
            body["pool"] = pool_id
        if namespace is not None:
            body["namespace"] = namespace
        response = get_session().post(
            f"{client.endpoint}/api/sandboxes",
            headers=api._build_mega_headers(token=client.token),
            json=body,
        )
        mega_raise_for_status(response)
        return cls(response.json()["sandbox"], endpoint=client.endpoint, token=client.token)

    @classmethod
    def connect(cls, sandbox_id: str, *, token: str | None = None, namespace: str | None = None) -> "NativeSandbox":
        client = MegaHubClient(token=token)
        api = MegaApi(endpoint=client.endpoint, token=client.token)
        response = get_session().get(
            f"{client.endpoint}/api/sandboxes/{sandbox_id}",
            headers=api._build_mega_headers(token=client.token),
        )
        mega_raise_for_status(response)
        data = response.json()["sandbox"]
        if namespace is not None and data.get("namespace") != namespace:
            raise SandboxError(f"Sandbox '{sandbox_id}' is not in namespace '{namespace}'.")
        return cls(data, endpoint=client.endpoint, token=client.token)

    @classmethod
    def list(cls, *, token: str | None = None, namespace: str | None = None) -> list[dict[str, Any]]:
        client = MegaHubClient(token=token)
        api = MegaApi(endpoint=client.endpoint, token=client.token)
        response = get_session().get(
            f"{client.endpoint}/api/sandboxes",
            headers=api._build_mega_headers(token=client.token),
            params={"namespace": namespace} if namespace else None,
        )
        mega_raise_for_status(response)
        return list(response.json().get("sandboxes", []))

    def kill(self) -> None:
        response = get_session().delete(f"{self._endpoint}/api/sandboxes/{self.id}", headers=self._headers)
        mega_raise_for_status(response)
        self.raw = response.json()["sandbox"]

    def close(self) -> None:
        """Native Sessions use short-lived HTTP requests and hold no client resources."""

    def run(
        self,
        cmd: str | list[str],
        *,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        stdin: str | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        check: bool = True,
        background: bool = False,
        shell: bool | None = None,
    ) -> SandboxCommandResult | NativeSandboxProcess:
        if background:
            return self._spawn(cmd, env=env, cwd=cwd)
        command = ["/bin/sh", "-lc", cmd] if isinstance(cmd, str) else list(cmd)
        response = get_session().post(
            f"{self._endpoint}/api/sandboxes/{self.id}/exec",
            headers={**self._headers, "Content-Type": "application/json"},
            json={
                "command": command,
                "cwd": cwd or "/workspace",
                "environment": {key: str(value) for key, value in (env or {}).items()},
                "timeoutSeconds": max(1, math.ceil(timeout)) if timeout is not None else 300,
                "stdin": stdin,
                "background": False,
            },
            stream=True,
        )
        mega_raise_for_status(response)
        stdout: list[str] = []
        stderr: list[str] = []
        exit_code: int | None = None
        duration_ms = 0
        timed_out = False
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "stdout":
                value = str(event.get("data") or "")
                stdout.append(value)
                if on_stdout:
                    on_stdout(value)
            elif event.get("type") == "stderr":
                value = str(event.get("data") or "")
                stderr.append(value)
                if on_stderr:
                    on_stderr(value)
            elif event.get("type") == "exit":
                exit_code = int(event.get("exit_code", 1))
                duration_ms = int(event.get("duration_ms", 0))
                timed_out = exit_code == 124
        result = SandboxCommandResult(exit_code=exit_code, stdout="".join(stdout), stderr="".join(stderr), timed_out=timed_out, duration_ms=duration_ms)
        if check and exit_code != 0:
            raise SandboxError(result.stderr or f"Command exited with status {exit_code}.")
        return result

    def _spawn(self, cmd: str | list[str], *, env: dict[str, Any] | None, cwd: str | None) -> NativeSandboxProcess:
        command = ["/bin/sh", "-lc", cmd] if isinstance(cmd, str) else list(cmd)
        launcher = (
            "import json,os,pathlib,subprocess,sys,time; "
            "root=pathlib.Path('/tmp/mega-processes'); root.mkdir(exist_ok=True); "
            "argv=json.loads(sys.argv[1]); extra=json.loads(sys.argv[2]); "
            "log=open(root/'process.log','ab',buffering=0); "
            "p=subprocess.Popen(argv,cwd=sys.argv[3],env={**os.environ,**extra},stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True); "
            "(root/f'{p.pid}.json').write_text(json.dumps({'pid':p.pid,'cmd':argv,'started_at_ms':int(time.time()*1000)})); print(p.pid)"
        )
        result = self.run(["python3", "-c", launcher, json.dumps(command), json.dumps({key: str(value) for key, value in (env or {}).items()}), cwd or "/workspace"])
        return NativeSandboxProcess(pid=int(result.stdout.strip()), cmd=cmd, _sandbox=self)

    def processes(self) -> list[NativeSandboxProcess]:
        script = (
            "import glob,json,os,pathlib; out=[]; "
            "[(lambda d: out.append({**d,'running':os.path.exists('/proc/'+str(d['pid']))}))(json.loads(pathlib.Path(p).read_text())) for p in glob.glob('/tmp/mega-processes/*.json')]; "
            "print(json.dumps(out))"
        )
        result = self.run(["python3", "-c", script], check=False)
        if result.exit_code != 0 or not result.stdout.strip():
            return []
        return [NativeSandboxProcess(pid=int(item["pid"]), cmd=item.get("cmd", ""), running=bool(item.get("running")), exit_code=None, _sandbox=self) for item in json.loads(result.stdout)]


def _volume_payload(volume: Any) -> dict[str, Any]:
    volume_type = getattr(volume, "type", None)
    if hasattr(volume_type, "value"):
        volume_type = volume_type.value
    return {
        "type": str(volume_type),
        "source": str(volume.source),
        "mount_path": str(volume.mount_path),
        "revision": str(volume.revision or "main"),
        "read_only": bool(volume.read_only),
        "path": volume.path,
    }


class NativeSandboxPool:
    """Native warm-template pool managed through ``/api/sandbox-pools``."""

    def __init__(self, data: dict[str, Any], *, endpoint: str, token: str | None) -> None:
        self.raw = data
        self.name = str(data["id"])
        self.id = self.name
        self.image = str(data.get("image") or "python:3.13")
        self.flavor = str(data.get("flavor") or "cpu-basic")
        self.host_ids = [self.name]
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._api = MegaApi(endpoint=endpoint, token=token)
        self._headers = self._api._build_mega_headers(token=token)

    @classmethod
    def create_pool(
        cls,
        *,
        image: str = "python:3.13",
        flavor: str = "cpu-basic",
        per_host: int = 4,
        max_hosts: int | None = None,
        idle_timeout: str | int = "1h",
        namespace: str | None = None,
        token: str | None = None,
    ) -> "NativeSandboxPool":
        client = MegaHubClient(token=token)
        api = MegaApi(endpoint=client.endpoint, token=client.token)
        body: dict[str, Any] = {
            "image": image,
            "flavor": flavor,
            "perHost": per_host,
            "maxHosts": max_hosts or 1,
            "idleTimeoutSeconds": idle_timeout,
        }
        if namespace is not None:
            body["namespace"] = namespace
        response = get_session().post(
            f"{client.endpoint}/api/sandbox-pools",
            headers=api._build_mega_headers(token=client.token),
            json=body,
        )
        mega_raise_for_status(response)
        return cls(response.json()["pool"], endpoint=client.endpoint, token=client.token)

    @classmethod
    def connect(cls, pool_id: str, *, namespace: str | None = None, token: str | None = None) -> "NativeSandboxPool":
        pools = cls.list(token=token, namespace=namespace)
        data = next((pool for pool in pools if pool.get("id") == pool_id), None)
        if data is None:
            raise SandboxError(f"Sandbox pool '{pool_id}' was not found.")
        client = MegaHubClient(token=token)
        return cls(data, endpoint=client.endpoint, token=client.token)

    @classmethod
    def list(cls, *, token: str | None = None, namespace: str | None = None) -> list[dict[str, Any]]:
        client = MegaHubClient(token=token)
        api = MegaApi(endpoint=client.endpoint, token=client.token)
        response = get_session().get(
            f"{client.endpoint}/api/sandbox-pools",
            headers=api._build_mega_headers(token=client.token),
            params={"namespace": namespace} if namespace else None,
        )
        mega_raise_for_status(response)
        return list(response.json().get("pools", []))

    def create(
        self,
        *,
        env: dict[str, str] | None = None,
        secrets: dict[str, str] | None = None,
        idle_timeout: str | int = "10m",
        max_lifetime: str | int = "1h",
        forward_mega_token: bool = False,
    ) -> NativeSandbox:
        environment = dict(env or {})
        if forward_mega_token:
            if not self._api.token:
                raise SandboxError("forward_mega_token requires authentication")
            environment["MEGA_TOKEN"] = self._api.token
        return NativeSandbox.create(
            runtime="python-3.13",
            flavor=self.flavor,
            idle_timeout=idle_timeout,
            max_lifetime=max_lifetime,
            env=environment,
            secrets=secrets,
            namespace=self.raw.get("namespace"),
            pool_id=self.id,
            token=self._token,
        )

    def delete(self) -> None:
        response = get_session().delete(f"{self._endpoint}/api/sandbox-pools/{self.id}", headers=self._headers)
        mega_raise_for_status(response)

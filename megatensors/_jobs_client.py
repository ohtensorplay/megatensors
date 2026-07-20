"""Native MEGA Jobs transport kept separate from the general Hub client."""

import http.client
import json
import time
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from megatensors._hub._jobs_api import JobHardwareInfo, JobInfo, ScheduledJobInfo
from megatensors._hub._space_api import Volume

JOB_FLAVOR = "cpu-nano"


class JobsClientMixin:
    """Typed Jobs methods mixed into :class:`megatensors.hub.MegaHubClient`."""

    def run_job(
        self,
        *,
        image: str,
        command: Iterable[str],
        arguments: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
        secrets: Mapping[str, str] | None = None,
        flavor: str = JOB_FLAVOR,
        timeout: int | str | None = None,
        labels: Mapping[str, str] | None = None,
        volumes: Iterable[Volume] | None = None,
        ssh: bool = False,
        namespace: str | None = None,
    ) -> JobInfo:
        """Submit one bounded native MEGA Job to the protected VPS runner."""
        data = self._request_json("POST", "/api/jobs", json_body=_job_payload(
            image=image,
            command=command,
            arguments=arguments,
            env=env,
            secrets=secrets,
            flavor=flavor,
            timeout=timeout,
            labels=labels,
            volumes=volumes,
            ssh=ssh,
            namespace=namespace,
        ), auth=True)
        return _job_info(data["job"], self.endpoint)

    create_job = run_job

    def list_jobs(
        self,
        *,
        limit: int = 30,
        stages: Iterable[str] | str | None = None,
        labels: Mapping[str, str] | None = None,
        namespace: str | None = None,
    ) -> list[JobInfo]:
        """List Jobs visible to the authenticated account."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        query: list[tuple[str, str]] = [("limit", str(limit))]
        if stages:
            raw_stages = [stages] if isinstance(stages, str) else stages
            selected = [str(stage).upper() for stage in raw_stages]
            if selected:
                query.append(("status", ",".join(selected)))
        if labels:
            query.extend(("label", f"{key}={value}") for key, value in _string_map(labels, "labels").items())
        if namespace:
            query.append(("namespace", namespace))
        data = self._request_json("GET", _query_path("/api/jobs", query), auth=True)
        return [_job_info(item, self.endpoint) for item in data.get("jobs", [])]

    def inspect_job(self, job_id: str) -> JobInfo:
        """Return current state for one Job."""
        data = self._request_json("GET", f"/api/jobs/{_quote_id(job_id)}", auth=True)
        return _job_info(data["job"], self.endpoint)

    get_job = inspect_job

    def cancel_job(self, job_id: str) -> JobInfo:
        """Request cancellation of a scheduling or running Job."""
        data = self._request_json("DELETE", f"/api/jobs/{_quote_id(job_id)}", auth=True)
        return _job_info(data["job"], self.endpoint)

    def fetch_job_logs(self, job_id: str, *, follow: bool = False, tail: int | None = None) -> Iterator[str]:
        """Yield SSE log lines from a retained Job container."""
        if tail is not None and (not isinstance(tail, int) or isinstance(tail, bool) or not 1 <= tail <= 5_000):
            raise ValueError("tail must be between 1 and 5000")
        query = [("follow", "true" if follow else "false")]
        if tail is not None:
            query.append(("tail", str(tail)))
        yield from _stream_sse(
            self,
            _query_path(f"/api/jobs/{_quote_id(job_id)}/logs", query),
            timeout=3_700 if follow else 120,
        )

    def fetch_job_metrics(
        self,
        job_id: str,
        *,
        namespace: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield live CPU, memory, and network metrics from a running Job.

        The Compute Pool owns the metrics stream. A namespace is required by
        the Hub route; when it is not supplied, resolve the Job's authoritative
        owner instead of guessing from the local identity.
        """
        if namespace is None:
            namespace = self.inspect_job(job_id).owner.name
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty handle")
        path = f"/api/jobs/{_quote_id(namespace)}/{_quote_id(job_id)}/metrics"
        for event in _stream_sse(self, path, timeout=15):
            try:
                metric = json.loads(event)
            except json.JSONDecodeError as error:
                raise ValueError("Job metrics stream returned invalid JSON") from error
            if not isinstance(metric, Mapping):
                raise ValueError("Job metrics stream must contain JSON objects")
            yield dict(metric)

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 2.0,
    ) -> JobInfo:
        """Poll one Job until it reaches a terminal state."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.inspect_job(job_id)
            if str(job.status.stage) in {"COMPLETED", "CANCELED", "ERROR", "DELETED"}:
                return job
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"Job '{job_id}' is still {job.status.stage} after {timeout} seconds")
            time.sleep(poll_interval if remaining is None else min(poll_interval, remaining))

    def list_jobs_hardware(self) -> list[JobHardwareInfo]:
        """List hardware flavors enabled by the native runner."""
        data = self._request_json("GET", "/api/jobs/hardware", auth=True)
        hardware = data if isinstance(data, list) else data.get("hardware", [])
        return [JobHardwareInfo(**item) for item in hardware]

    def get_jobs_usage(self, *, namespace: str | None = None) -> dict[str, Any]:
        """Return per-minute Jobs usage and accrued USD cost for the current billing owner."""
        query = [("namespace", namespace)] if namespace else []
        return dict(self._request_json("GET", _query_path("/api/jobs/usage", query), auth=True))

    def get_compute_billing(self, *, namespace: str | None = None) -> dict[str, Any]:
        """Return prepaid compute balance and immutable ledger for a personal or organization owner."""
        query = [("organization", namespace)] if namespace else []
        data = self._request_json("GET", _query_path("/api/billing/compute", query), auth=True)
        compute = data.get("compute", {})
        if not isinstance(compute, Mapping):
            raise ValueError("compute billing response must contain an object")
        return dict(compute)

    def create_scheduled_job(
        self,
        *,
        image: str,
        command: Iterable[str],
        schedule: str,
        arguments: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
        secrets: Mapping[str, str] | None = None,
        flavor: str = JOB_FLAVOR,
        timeout: int | str | None = None,
        labels: Mapping[str, str] | None = None,
        volumes: Iterable[Volume] | None = None,
        ssh: bool = False,
        namespace: str | None = None,
        suspend: bool = False,
        concurrency: bool = False,
    ) -> ScheduledJobInfo:
        """Create a UTC cron Job schedule owned by the authenticated account."""
        if not isinstance(schedule, str) or not schedule.strip():
            raise ValueError("schedule must be a non-empty cron expression or alias")
        payload = _job_payload(
            image=image,
            command=command,
            arguments=arguments,
            env=env,
            secrets=secrets,
            flavor=flavor,
            timeout=timeout,
            labels=labels,
            volumes=volumes,
            ssh=ssh,
            namespace=namespace,
        )
        payload.update({"schedule": schedule, "suspend": bool(suspend), "concurrency": bool(concurrency)})
        data = self._request_json("POST", "/api/jobs/scheduled", json_body=payload, auth=True)
        return ScheduledJobInfo(**data["scheduledJob"])

    def list_scheduled_jobs(self, *, namespace: str | None = None) -> list[ScheduledJobInfo]:
        """List recurring Jobs visible to the authenticated account."""
        path = _query_path("/api/jobs/scheduled", [("namespace", namespace)] if namespace else [])
        data = self._request_json("GET", path, auth=True)
        return [ScheduledJobInfo(**item) for item in data.get("scheduledJobs", [])]

    def inspect_scheduled_job(self, schedule_id: str) -> ScheduledJobInfo:
        """Return one recurring Job schedule."""
        data = self._request_json("GET", f"/api/jobs/scheduled/{_quote_id(schedule_id)}", auth=True)
        return ScheduledJobInfo(**data["scheduledJob"])

    def delete_scheduled_job(self, schedule_id: str) -> None:
        """Delete one recurring Job schedule without affecting prior runs."""
        self._request("DELETE", f"/api/jobs/scheduled/{_quote_id(schedule_id)}", auth=True)

    def suspend_scheduled_job(self, schedule_id: str) -> ScheduledJobInfo:
        """Pause dispatch for one recurring Job schedule."""
        return self._set_scheduled_job_state(schedule_id, "suspend")

    def resume_scheduled_job(self, schedule_id: str) -> ScheduledJobInfo:
        """Resume dispatch for one recurring Job schedule."""
        return self._set_scheduled_job_state(schedule_id, "resume")

    def trigger_scheduled_job(self, schedule_id: str) -> JobInfo:
        """Dispatch an immediate run from a saved schedule."""
        data = self._request_json("POST", f"/api/jobs/scheduled/{_quote_id(schedule_id)}/trigger", auth=True)
        return _job_info(data["job"], self.endpoint)

    def _set_scheduled_job_state(self, schedule_id: str, action: str) -> ScheduledJobInfo:
        data = self._request_json("POST", f"/api/jobs/scheduled/{_quote_id(schedule_id)}/{action}", auth=True)
        return ScheduledJobInfo(**data["scheduledJob"])


def _job_payload(
    *,
    image: str,
    command: Iterable[str],
    arguments: Iterable[str] | None,
    env: Mapping[str, str] | None,
    secrets: Mapping[str, str] | None,
    flavor: str,
    timeout: int | str | None,
    labels: Mapping[str, str] | None,
    volumes: Iterable[Volume] | None,
    ssh: bool,
    namespace: str | None,
) -> dict[str, Any]:
    if not isinstance(image, str) or not image.strip():
        raise ValueError("image must be a non-empty Docker image reference")
    command_items = _strings(command, "command")
    if not command_items:
        raise ValueError("command must contain at least one argument")
    payload: dict[str, Any] = {
        "dockerImage": image,
        "command": command_items,
        "arguments": _strings(arguments or (), "arguments"),
        "environment": _string_map(env or {}, "environment"),
        "secrets": _string_map(secrets or {}, "secrets"),
        "flavor": flavor,
        "labels": _string_map(labels or {}, "labels"),
    }
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, str)):
            raise ValueError("timeout must be an integer number of seconds or a duration string")
        payload["timeoutSeconds"] = timeout
    volume_items = list(volumes or ())
    if any(not isinstance(volume, Volume) for volume in volume_items):
        raise ValueError("volumes must contain Volume instances")
    if volume_items:
        payload["volumes"] = [volume.to_dict() for volume in volume_items]
    if not isinstance(ssh, bool):
        raise ValueError("ssh must be a boolean")
    if ssh:
        payload["ssh"] = {"enabled": True}
    if namespace is not None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty handle")
        payload["namespace"] = namespace
    overlap = set(payload["environment"]) & set(payload["secrets"])
    if overlap:
        raise ValueError(f"environment and secrets both define {sorted(overlap)[0]}")
    return payload


def _strings(values: Iterable[str], name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of strings, not one string")
    result = list(values)
    if any(not isinstance(value, str) or "\x00" in value for value in result):
        raise ValueError(f"{name} must contain strings without NUL bytes")
    return result


def _string_map(values: Mapping[str, str], name: str) -> dict[str, str]:
    result = dict(values)
    if any(not isinstance(key, str) or not isinstance(value, str) or "\x00" in key or "\x00" in value for key, value in result.items()):
        raise ValueError(f"{name} must be a string map without NUL bytes")
    return result


def _job_info(data: Mapping[str, Any], endpoint: str) -> JobInfo:
    job = JobInfo(**{**data, "endpoint": endpoint})
    if isinstance(data.get("url"), str):
        job.url = str(data["url"])
    return job


def _query_path(path: str, query: Iterable[tuple[str, str]]) -> str:
    values = [(key, value) for key, value in query if value is not None]
    return f"{path}?{urllib.parse.urlencode(values)}" if values else path


def _quote_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Job id must be a non-empty string")
    return urllib.parse.quote(value, safe="")


def _stream_sse(client: Any, path: str, *, timeout: float) -> Iterator[str]:
    from megatensors.hub import MegaHubError

    if not client.token:
        raise MegaHubError("not logged in; run `mega auth login --token ...` first")
    url = urllib.parse.urlsplit(client.endpoint + path)
    target = url.path or "/"
    if url.query:
        target += f"?{url.query}"
    conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(url.netloc, timeout=timeout)
    try:
        conn.request("GET", target, headers={"Accept": "text/event-stream", "Authorization": f"Bearer {client.token}"})
        response = conn.getresponse()
        if response.status >= 400:
            raw = response.read()
            message = response.reason
            try:
                payload = json.loads(raw.decode("utf-8"))
                message = payload.get("error") or payload.get("message") or message
            except Exception:
                pass
            raise MegaHubError(f"{response.status} {message}", status_code=response.status, method="GET", url=client.endpoint + path)
        while raw := response.readline():
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data:"):
                yield line[5:].lstrip()
    except OSError as error:
        raise MegaHubError(str(error), method="GET", url=client.endpoint + path) from error
    finally:
        conn.close()

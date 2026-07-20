from __future__ import annotations

import json
from typing import Any

import pytest

from megatensors.hub import MegaHubClient
from megatensors._hub._space_api import Volume


def _job_payload(job_id: str = "job-0123456789abcdef0123456789abcdef") -> dict[str, Any]:
    return {
        "id": job_id,
        "createdAt": "2026-07-13T00:00:00.000Z",
        "startedAt": None,
        "finishedAt": None,
        "dockerImage": "python:3.12-slim",
        "command": ["python", "-c", "print('ok')"],
        "arguments": [],
        "environment": {"MODE": "test"},
        "secrets": {},
        "secretsConfigured": True,
        "flavor": "cpu-nano",
        "labels": {"lane": "contract"},
        "status": {"stage": "SCHEDULING", "message": None},
        "durations": {"schedulingSecs": None, "runningSecs": None, "totalSecs": 0},
        "billing": {
            "unitCostMicroUSD": 167,
            "unitCostUSD": 0.000167,
            "unitLabel": "minute",
            "billedMinutes": 0,
            "amountMicroUSD": 0,
            "amountUSD": 0,
        },
        "owner": {"id": "alice@example.test", "name": "alice", "type": "user"},
        "initiator": {"type": "user", "id": "alice@example.test", "name": "alice"},
        "url": "https://hub.example.test/settings/jobs?job=job-0123456789abcdef0123456789abcdef",
    }


def _schedule_payload() -> dict[str, Any]:
    return {
        "id": "schedule-0123456789abcdef0123456789abcdef",
        "createdAt": "2026-07-13T00:00:00.000Z",
        "schedule": "0 * * * *",
        "suspend": False,
        "concurrency": False,
        "owner": {"id": "alice@example.test", "name": "alice", "type": "user"},
        "jobSpec": {
            "dockerImage": "python:3.12-slim",
            "command": ["python", "-c", "print('ok')"],
            "arguments": [],
            "environment": {"MODE": "test"},
            "secrets": {},
            "flavor": "cpu-nano",
            "timeout": 300,
            "labels": {"lane": "contract"},
            "volumes": None,
        },
        "status": {"lastJob": None, "nextJobRunAt": "2026-07-13T01:00:00.000Z"},
        "url": "https://hub.example.test/settings/jobs?scheduled=schedule-0123456789abcdef0123456789abcdef",
    }


def test_native_jobs_client_uses_worker_routes_and_typed_responses(monkeypatch):
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class RecordingClient(MegaHubClient):
        def _request_json(self, method, path, **kwargs):  # type: ignore[override]
            calls.append((method, path, kwargs))
            if path == "/api/jobs/hardware":
                return [{
                    "name": "cpu-nano",
                    "prettyName": "CPU Nano",
                    "cpu": "0.25 vCPU",
                    "ram": "256 MB RAM + 256 MB swap",
                    "ephemeralStorage": "10 GB",
                    "accelerator": None,
                    "unitCostMicroUSD": 167,
                    "unitCostUSD": 0.000167,
                    "unitLabel": "minute",
                }]
            if path == "/api/jobs/usage":
                return {"currency": "USD", "billedMinutes": 2, "amountUSD": 0.000334, "jobs": []}
            if path == "/api/billing/compute":
                return {"compute": {"currency": "USD", "balanceMicroUSD": 10_000_000, "balanceUSD": 10}}
            if path.endswith("/scheduled") and method == "POST":
                return {"scheduledJob": _schedule_payload()}
            if path.endswith("/scheduled") and method == "GET":
                return {"scheduledJobs": [_schedule_payload()]}
            if method == "GET":
                return {"jobs": [_job_payload()]}
            return {"job": _job_payload()}

    client = RecordingClient(endpoint="https://hub.example.test", token="token")
    job = client.run_job(
        image="python:3.12-slim",
        command=["python", "-c", "print('ok')"],
        env={"MODE": "test"},
        secrets={"API_TOKEN": "sealed"},
        labels={"lane": "contract"},
        volumes=[Volume(type="bucket", source="alice/artifacts", mount_path="/output", read_only=False)],
        timeout="5m",
        namespace="research",
    )

    assert job.id == _job_payload()["id"]
    assert job.url.startswith("https://hub.example.test/settings/jobs")
    assert calls[0] == (
        "POST",
        "/api/jobs",
        {
            "json_body": {
                "dockerImage": "python:3.12-slim",
                "command": ["python", "-c", "print('ok')"],
                "arguments": [],
                "environment": {"MODE": "test"},
                "secrets": {"API_TOKEN": "sealed"},
                "flavor": "cpu-nano",
                "labels": {"lane": "contract"},
                "volumes": [{"type": "bucket", "source": "alice/artifacts", "mountPath": "/output", "readOnly": False}],
                "timeoutSeconds": "5m",
                "namespace": "research",
            },
            "auth": True,
        },
    )

    assert client.list_jobs_hardware()[0].ram == "256 MB RAM + 256 MB swap"
    assert client.get_jobs_usage()["amountUSD"] == 0.000334
    assert client.get_compute_billing()["balanceUSD"] == 10
    assert job.billing is not None
    assert job.billing.unit_cost_micro_usd == 167
    scheduled = client.create_scheduled_job(
        image="python:3.12-slim",
        command=["python", "-c", "print('ok')"],
        schedule="@hourly",
    )
    assert scheduled.schedule == "0 * * * *"
    assert len(client.list_scheduled_jobs()) == 1

    streamed: list[tuple[str, int]] = []

    def fake_stream(_client, path, *, timeout):
        streamed.append((path, timeout))
        if path.endswith("/metrics"):
            return iter([json.dumps({"cpu_usage_pct": 12.5, "gpus": {}})])
        return iter(["first line", "second line"])

    monkeypatch.setattr("megatensors._jobs_client._stream_sse", fake_stream)
    assert list(client.fetch_job_logs(job.id, follow=True, tail=20)) == ["first line", "second line"]
    assert list(client.fetch_job_metrics(job.id, namespace="research")) == [{"cpu_usage_pct": 12.5, "gpus": {}}]
    assert streamed == [
        (f"/api/jobs/{job.id}/logs?follow=true&tail=20", 3_700),
        (f"/api/jobs/research/{job.id}/metrics", 15),
    ]


def test_native_jobs_client_keeps_single_status_and_command_as_single_values():
    calls: list[tuple[str, str]] = []

    class RecordingClient(MegaHubClient):
        def _request_json(self, method, path, **_kwargs):  # type: ignore[override]
            calls.append((method, path))
            return {"jobs": []}

    client = RecordingClient(endpoint="https://hub.example.test", token="token")
    assert client.list_jobs(stages="RUNNING") == []
    assert calls == [("GET", "/api/jobs?limit=30&status=RUNNING")]
    with pytest.raises(ValueError, match="not one string"):
        client.run_job(image="python:3.12-slim", command="python -V")

from __future__ import annotations

import inspect

import megatensors
import megatensors._hub as hub
import megatensors.mega_hub as mega_hub
from megatensors._hub._jobs_api import _create_job_spec


def test_hf_hub_public_aliases_are_backed_by_mega_implementations():
    """Compatibility imports must execute MEGA code, never proxy HF services."""

    expected = {
        "HFCacheInfo",
        "HFSummaryWriter",
        "HUGGINGFACE_CO_URL_HOME",
        "HUGGINGFACE_CO_URL_TEMPLATE",
        "HfApi",
        "HfFileMetadata",
        "HfFileSystem",
        "HfFileSystemFile",
        "HfFileSystemResolvedPath",
        "HfFileSystemStreamFile",
        "HfUri",
        "attach_huggingface_oauth",
        "check_cli_update",
        "get_hf_file_metadata",
        "hf_hub_download",
        "hf_hub_url",
        "hf_raise_for_status",
        "parse_hf_uri",
        "parse_huggingface_oauth",
        "repo_type_and_id_from_hf_id",
        "typer_factory",
    }

    assert expected <= set(hub.__all__)
    assert hub.HfApi is hub.MegaApi
    assert hub.HfFileSystem is hub.MegaFileSystem
    assert hub.HfFileMetadata is hub.MegaFileMetadata
    assert hub.HFCacheInfo is hub.MegaCacheInfo
    assert hub.HFSummaryWriter is hub.MegaSummaryWriter
    assert hub.HfUri is hub.MegaUri
    assert hub.hf_hub_download is hub.mega_hub_download
    assert hub.hf_hub_url is hub.mega_hub_url
    assert hub.get_hf_file_metadata is hub.get_mega_file_metadata
    assert hub.hf_raise_for_status is hub.mega_raise_for_status
    assert hub.parse_hf_uri is hub.parse_mega_uri
    assert hub.attach_huggingface_oauth is hub.attach_mega_oauth
    assert hub.parse_huggingface_oauth is hub.parse_mega_oauth
    assert hub.repo_type_and_id_from_hf_id is hub.repo_type_and_id_from_mega_id


def test_hf_hub_aliases_keep_hub_signatures_and_mega_transport():
    assert inspect.signature(hub.hf_hub_download) == inspect.signature(hub.mega_hub_download)
    assert inspect.signature(hub.hf_hub_url) == inspect.signature(hub.mega_hub_url)
    assert inspect.signature(hub.HfApi) == inspect.signature(hub.MegaApi)
    assert hub.HUGGINGFACE_CO_URL_HOME == "https://mega.tensorplay.cn/"
    assert hub.hf_hub_url("org/repo", "config.json") == (
        "https://mega.tensorplay.cn/api/repos/org/repo/resolve/config.json?revision=main"
    )


def test_common_hf_hub_imports_are_available_from_sdk_entrypoints():
    assert megatensors.HfApi is hub.HfApi
    assert megatensors.HfFileSystem is hub.HfFileSystem
    assert megatensors.hf_hub_download is hub.hf_hub_download
    assert megatensors.hf_hub_url is hub.hf_hub_url
    assert mega_hub.HfApi is hub.HfApi
    assert mega_hub.hf_hub_download is hub.hf_hub_download


def test_sdk_encodes_hf_job_ssh_for_the_real_runner_ingress():
    spec = _create_job_spec(
        image="python:3.12-slim",
        command=["python", "-c", "print('hello')"],
        env=None,
        secrets=None,
        flavor=None,
        timeout=None,
        ssh=True,
    )
    assert spec["ssh"] == {"enabled": True}


def test_scheduled_uv_jobs_forward_ssh_to_the_real_scheduled_job_spec(monkeypatch):
    api = hub.MegaApi(endpoint="https://hub.example.test", token="token")
    calls = []

    monkeypatch.setattr(
        api,
        "_create_uv_command_env_and_secrets",
        lambda **_kwargs: (["uv", "run", "https://example.test/task.py"], {}, {}, []),
    )
    monkeypatch.setattr(api, "create_scheduled_job", lambda **kwargs: calls.append(kwargs) or object())

    api.create_scheduled_uv_job(
        "https://example.test/task.py",
        schedule="@hourly",
        ssh=True,
        namespace="research",
    )

    assert calls == [{
        "image": "ghcr.io/astral-sh/uv:python3.12-bookworm",
        "command": ["uv", "run", "https://example.test/task.py"],
        "schedule": "@hourly",
        "suspend": None,
        "concurrency": None,
        "env": {},
        "secrets": {},
        "flavor": None,
        "timeout": None,
        "labels": None,
        "volumes": None,
        "expose": None,
        "ssh": True,
        "namespace": "research",
        "token": None,
    }]

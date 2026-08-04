# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import io

import httpx

from megatensors._hub import constants, lfs
from megatensors._hub.utils import _git_credential


def _capture_batch_url(monkeypatch, *, endpoint: str | None, repo_type: str = "dataset") -> str:
    requested: list[str] = []

    def fake_backoff(method: str, url: str, **kwargs):
        requested.append(url)
        return httpx.Response(200, json={"objects": []}, request=httpx.Request(method, url))

    monkeypatch.setattr(lfs, "http_backoff", fake_backoff)
    lfs.post_lfs_batch_info(
        upload_infos=[],
        token=None,
        repo_type=repo_type,
        repo_id="mega/example",
        endpoint=endpoint,
    )
    return requested[0]


def test_production_lfs_uses_dedicated_git_origin_without_type_prefix(monkeypatch):
    monkeypatch.setattr(constants, "ENDPOINT", "https://mega.tensorplay.cn")
    monkeypatch.setattr(constants, "GIT_ENDPOINT", "https://git.tensorplay.cn")

    assert _capture_batch_url(monkeypatch, endpoint=constants.ENDPOINT) == (
        "https://git.tensorplay.cn/mega/example.git/info/lfs/objects/batch"
    )


def test_self_hosted_lfs_keeps_the_hub_origin_and_type_prefix(monkeypatch):
    monkeypatch.setattr(constants, "ENDPOINT", "https://mega.tensorplay.cn")
    monkeypatch.setattr(constants, "GIT_ENDPOINT", "https://git.tensorplay.cn")

    assert _capture_batch_url(monkeypatch, endpoint="https://hub.example.test") == (
        "https://hub.example.test/datasets/mega/example.git/info/lfs/objects/batch"
    )


def test_git_credentials_cover_hub_and_redirected_git_origins(monkeypatch):
    writes: list[tuple[str, str]] = []

    @contextmanager
    def fake_interactive(command: str, *, folder: str | None = None):
        stdin = io.StringIO()
        yield stdin, io.StringIO()
        writes.append((command, stdin.getvalue()))

    monkeypatch.setattr(constants, "ENDPOINT", "https://mega.tensorplay.cn")
    monkeypatch.setattr(constants, "GIT_ENDPOINT", "https://git.tensorplay.cn")
    monkeypatch.setattr(_git_credential, "run_interactive_subprocess", fake_interactive)

    _git_credential.set_git_credential("token", username="Mega_User")
    _git_credential.unset_git_credential(username="Mega_User")

    assert writes == [
        ("git credential approve", "url=https://mega.tensorplay.cn\nusername=mega_user\npassword=token\n\n"),
        ("git credential approve", "url=https://git.tensorplay.cn\nusername=mega_user\npassword=token\n\n"),
        ("git credential reject", "url=https://mega.tensorplay.cn\nusername=mega_user\n\n"),
        ("git credential reject", "url=https://git.tensorplay.cn\nusername=mega_user\n\n"),
    ]

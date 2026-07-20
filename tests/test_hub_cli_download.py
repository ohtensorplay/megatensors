from __future__ import annotations

import io
from pathlib import Path

import httpx
import pytest

from megatensors._hub import constants
from megatensors._hub.cli._output import OutputFormat, out
from megatensors._hub.cli.download import download
from megatensors._hub.errors import CLIError
from megatensors._hub.file_download import http_get, mega_hub_url


def test_worker_resolve_url_keeps_dataset_repo_id_two_segments():
    assert mega_hub_url("mega/demo", "data/train.jsonl", repo_type="dataset") == (
        "https://mega.tensorplay.cn/api/repos/mega/demo/resolve/data/train.jsonl?revision=main"
    )


def test_download_uses_mega_uri_and_worker_repo_id(monkeypatch):
    calls = []

    def fake_download_file(**kwargs):
        calls.append(kwargs)
        return "/tmp/data.jsonl"

    monkeypatch.setattr(
        "megatensors._hub.cli.download._download_file", fake_download_file
    )

    download("mega://datasets/mega/demo@release/data.jsonl")

    assert calls == [
        {
            "repo_id": "mega/demo",
            "repo_type": "dataset",
            "revision": "release",
            "filename": "data.jsonl",
            "cache_dir": None,
            "force_download": False,
            "token": None,
            "local_dir": None,
            "library_name": "mega-cli",
            "dry_run": False,
        }
    ]


def test_download_rejects_non_mega_uri():
    with pytest.raises(CLIError, match="mega://"):
        download("legacy://datasets/mega/demo")


def test_download_human_output_reports_file_status(monkeypatch, capsys):
    previous_mode = out.mode
    out.set_mode(OutputFormat.human)

    def fake_download_file(**kwargs):
        return "/tmp/config.json"

    monkeypatch.setattr(
        "megatensors._hub.cli.download._download_file", fake_download_file
    )
    try:
        download("mega/demo", ["config.json"])
    finally:
        out.set_mode(previous_mode)

    captured = capsys.readouterr()
    expected_dir = Path(constants.MEGA_HUB_CACHE).expanduser().resolve()
    assert captured.err == (
        f"Downloading Model file from {constants.ENDPOINT} to directory: {expected_dir}\n"
    )
    assert "✓ File downloaded" in captured.out
    assert "path: /tmp/config.json" in captured.out


def test_download_json_output_keeps_status_off_stderr(monkeypatch, capsys):
    previous_mode = out.mode
    out.set_mode(OutputFormat.json)

    def fake_download_file(**kwargs):
        return "/tmp/config.json"

    monkeypatch.setattr(
        "megatensors._hub.cli.download._download_file", fake_download_file
    )
    try:
        download("mega/demo", ["config.json"])
    finally:
        out.set_mode(previous_mode)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == '{"path": "/tmp/config.json"}\n'


def test_download_json_snapshot_keeps_per_file_progress_off(monkeypatch, capsys):
    previous_mode = out.mode
    out.set_mode(OutputFormat.json)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/demo"

    monkeypatch.setattr(
        "megatensors._hub.cli.download.snapshot_download", fake_snapshot_download
    )
    try:
        download("mega/demo")
    finally:
        out.set_mode(previous_mode)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == '{"path": "/tmp/demo"}\n'
    assert calls[0]["per_file_progress"] is False


def test_download_human_output_reports_snapshot_status(monkeypatch, capsys):
    previous_mode = out.mode
    out.set_mode(OutputFormat.human)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/demo"

    monkeypatch.setattr(
        "megatensors._hub.cli.download.snapshot_download", fake_snapshot_download
    )
    try:
        download("mega/demo", include=["*.json"], exclude=["*.bin"])
    finally:
        out.set_mode(previous_mode)

    captured = capsys.readouterr()
    expected_dir = Path(constants.MEGA_HUB_CACHE).expanduser().resolve()
    assert captured.err == (
        f"Downloading Model from {constants.ENDPOINT} to directory: {expected_dir}\n"
    )
    assert "✓ Snapshot ready" in captured.out
    assert calls[0]["per_file_progress"] is True


def test_http_get_progress_uses_modelscope_style_filename(monkeypatch):
    descs = []

    class FakeProgress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def update(self, n):
            return None

    def fake_progress_context(**kwargs):
        descs.append(kwargs["desc"])
        return FakeProgress()

    class FakeStream:
        def __enter__(self):
            request = httpx.Request("GET", "https://example.test/config.json")
            return httpx.Response(
                200, headers={"Content-Length": "3"}, content=b"abc", request=request
            )

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        "megatensors._hub.file_download._get_progress_bar_context",
        fake_progress_context,
    )
    monkeypatch.setattr(
        "megatensors._hub.file_download.http_stream_backoff", lambda **_: FakeStream()
    )

    target = io.BytesIO()
    http_get(
        "https://example.test/config.json",
        target,
        displayed_filename="config.json",
        expected_size=3,
    )

    assert target.getvalue() == b"abc"
    assert descs == ["Downloading [config.json]"]

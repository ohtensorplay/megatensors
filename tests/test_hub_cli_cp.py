# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import pytest

from megatensors._hub.cli import _cp
from megatensors._hub.cli._cli_utils import typer_factory
from megatensors._hub.errors import CLIError
from megatensors.hub import MegaHubClient


def _cp_app():
    app = typer_factory(help="MEGA copy test app")
    app.command()(_cp.make_cp())
    return app


def test_parse_mega_uri_requires_a_repository_file_and_uses_mega_protocol():
    uri = _cp.parse_mega_uri("mega://datasets/mega/data@release%2F2026/train/part-0.jsonl")

    assert uri.repo_id == "mega/data"
    assert uri.repo_type == "dataset"
    assert uri.revision == "release/2026"
    assert uri.path == "train/part-0.jsonl"
    assert uri.to_uri() == "mega://datasets/mega/data@release%2F2026/train/part-0.jsonl"

    root = _cp.parse_mega_uri("mega://models/mega/demo/")
    assert root.path == ""
    assert root.trailing_slash is True

    with pytest.raises(CLIError, match="mega://"):
        _cp.parse_mega_uri("invalid://mega/data/file.json")
    bucket = _cp.parse_mega_uri("mega://buckets/mega/archive/file.json")
    assert bucket.repo_id == "mega/archive"
    assert bucket.repo_type == "bucket"
    assert bucket.revision is None
    assert bucket.path == "file.json"
    assert bucket.to_uri() == "mega://buckets/mega/archive/file.json"


def test_cp_remote_to_remote_calls_the_worker_copy_operation(monkeypatch):
    calls = []

    class RecordingApi:
        def __init__(self, endpoint=None, token=None):
            calls.append(("init", endpoint, token))

        def copy_files(self, *args, **kwargs):
            calls.append(("copy_files", args, kwargs))
            return {"revision": "copied"}

    monkeypatch.setattr(_cp, "MegaHubClient", RecordingApi)
    result = CliRunner().invoke(
        _cp_app(),
        [
            "cp",
            "mega://models/mega/source@release/model.mega",
            "mega://spaces/mega/demo@main/assets/model.mega",
            "--token",
            "secret",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("init", None, "secret"),
        (
            "copy_files",
            ("mega/source", "model.mega", "mega/demo", "assets/model.mega"),
            {
                "source_revision": "release",
                "revision": "main",
                "source_merge_contents": False,
                "destination_is_directory": False,
            },
        ),
    ]
    assert "hf://" not in result.output
    assert "Copied" in result.output


def test_cp_preserves_directory_copy_semantics(monkeypatch):
    calls = []

    class RecordingApi:
        def __init__(self, endpoint=None, token=None):
            pass

        def copy_files(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"revision": "copied"}

    monkeypatch.setattr(_cp, "MegaHubClient", RecordingApi)
    result = CliRunner().invoke(
        _cp_app(),
        ["cp", "mega://models/mega/source@main/data/", "mega://datasets/mega/destination@main/copied/"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            ("mega/source", "data", "mega/destination", "copied"),
            {
                "source_revision": "main",
                "revision": "main",
                "source_merge_contents": True,
                "destination_is_directory": True,
            },
        )
    ]


def test_cp_local_file_upload_uses_destination_repository_type(tmp_path, monkeypatch):
    source = tmp_path / "data.jsonl"
    source.write_text('{"id": 1}\n', encoding="utf-8")
    calls = []

    class RecordingApi:
        def __init__(self, endpoint=None, token=None):
            pass

        def upload_file(self, repo_id, local_path, **kwargs):
            calls.append((repo_id, Path(local_path), kwargs))
            return {"revision": "uploaded"}

    monkeypatch.setattr(_cp, "MegaHubClient", RecordingApi)
    result = CliRunner().invoke(
        _cp_app(),
        ["cp", str(source), "mega://datasets/mega/data@main/train/data.jsonl"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "mega/data",
            source,
            {
                "path_in_repo": "train/data.jsonl",
                "revision": "main",
                "repo_type": "dataset",
            },
        )
    ]


def test_cp_supports_standard_input_and_standard_output(monkeypatch):
    uploaded = []

    class RecordingApi:
        def __init__(self, endpoint=None, token=None):
            pass

        def upload_file(self, repo_id, local_path, **kwargs):
            uploaded.append((repo_id, Path(local_path).read_bytes(), kwargs))
            return {"revision": "uploaded"}

        def download_file(self, repo_id, filename, *, local_dir, revision, force):
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"downloaded-bytes")
            return target

    monkeypatch.setattr(_cp, "MegaHubClient", RecordingApi)
    runner = CliRunner()
    upload = runner.invoke(
        _cp_app(),
        ["cp", "-", "mega://models/mega/demo@main/config.json"],
        input=b'{"source": "stdin"}\n',
    )
    download = runner.invoke(
        _cp_app(),
        ["cp", "mega://models/mega/demo@main/config.json", "-"],
    )

    assert upload.exit_code == 0, upload.output
    assert uploaded == [
        (
            "mega/demo",
            b'{"source": "stdin"}\n',
            {"path_in_repo": "config.json", "revision": "main", "repo_type": "model"},
        )
    ]
    assert download.exit_code == 0, download.output
    assert download.output_bytes == b"downloaded-bytes"


def test_hub_api_copy_file_posts_source_and_destination_metadata(monkeypatch):
    captured = {}

    def record_request(self, method, path, **kwargs):
        captured.update(method=method, path=path, **kwargs)
        return {"revision": "copied"}

    monkeypatch.setattr(MegaHubClient, "_request_json", record_request)
    api = MegaHubClient(endpoint="https://hub.example.test", token="secret")

    result = api.copy_file(
        "mega/source",
        "weights/model.mega",
        "mega/destination",
        "archive/model.mega",
        source_revision="release/2026",
        revision="main",
    )

    assert result == {"revision": "copied"}
    assert captured == {
        "method": "POST",
        "path": "/api/repos/mega/destination/copy",
        "json_body": {
            "source_repo_id": "mega/source",
            "source_path": "weights/model.mega",
            "source_revision": "release/2026",
            "path": "archive/model.mega",
            "revision": "main",
            "source_merge_contents": False,
            "destination_is_directory": False,
            "commit_message": "Copy mega/source/weights/model.mega",
        },
        "auth": True,
    }


def test_bucket_uri_upload_uses_native_bucket_batch(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}\n', encoding="utf-8")
    calls = []

    class RecordingBucketApi:
        def __init__(self, endpoint=None, token=None):
            calls.append(("init", endpoint, token))

        def batch_bucket_files(self, bucket_id, *, add):
            calls.append(("batch_bucket_files", bucket_id, add))

    monkeypatch.setattr(_cp, "MegaApi", RecordingBucketApi)
    command = _cp.make_cp()
    command(str(source), "mega://buckets/mega/demo/file.json", token="secret")

    assert calls[0] == ("init", None, "secret")
    assert calls[1][0:2] == ("batch_bucket_files", "mega/demo")
    assert calls[1][2] == [(source, "file.json")]


def test_contextual_cp_aliases_reject_the_other_resource_family():
    with pytest.raises(CLIError, match="mega buckets cp.*only works with Buckets"):
        _cp.make_cp("buckets")(
            "mega://models/mega/demo@main/config.json",
            "./config.json",
            token=None,
        )
    with pytest.raises(CLIError, match="mega repos cp.*only works with repositories"):
        _cp.make_cp("repos")(
            "mega://buckets/mega/demo/config.json",
            "./config.json",
            token=None,
        )

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from megatensors._hub.cli import upload as upload_module
from megatensors._hub.cli._cli_utils import RepoType
from megatensors._hub.cli.mega import app
from megatensors._hub.cli.upload import upload


def test_upload_forwards_mega_native_options(tmp_path, monkeypatch):
    calls = []

    def fake_run_upload(repo_id, **kwargs):
        calls.append((repo_id, kwargs))

    monkeypatch.setattr("megatensors._hub.cli.upload.run_upload", fake_run_upload)

    source = Path(tmp_path) / "weights"
    upload(
        "mega/demo",
        source,
        "checkpoint",
        repo_type=RepoType.dataset,
        revision="release",
        private=True,
        commit_message="publish dataset",
        commit_description="Dataset release notes",
        include=["*.jsonl"],
        exclude=["*.tmp"],
        max_workers=3,
        token="mega-token",
    )

    assert calls == [
        (
            "mega/demo",
            {
                "local_path": source,
                "path_in_repo": "checkpoint",
                "repo_type": "dataset",
                "revision": "release",
                "private": True,
                "commit_message": "publish dataset",
                "commit_description": "Dataset release notes",
                "create_pr": False,
                "include": ["*.jsonl"],
                "exclude": ["*.tmp"],
                "delete": None,
                "max_workers": 3,
                "sync": False,
                "token": "mega-token",
            },
        )
    ]


def test_upload_uses_shared_structured_output(tmp_path, monkeypatch):
    source = tmp_path / "weights.mega"
    source.write_bytes(b"weights")

    class FakeCommitInfo:
        oid = "rev-1"

    class FakeMegaApi:
        def __init__(self, *, token=None, library_name=None):
            assert token == "mega-token"
            assert library_name == "mega-cli"

        def create_repo(self, repo_id, **kwargs):
            assert repo_id == "mega/demo"
            assert kwargs == {"repo_type": "model", "private": False, "exist_ok": True}

        def upload_file(self, **kwargs):
            assert kwargs["repo_id"] == "mega/demo"
            assert kwargs["path_or_fileobj"] == source
            assert kwargs["path_in_repo"] == "weights.mega"
            assert kwargs["commit_description"] == "A compact release note."
            return FakeCommitInfo()

    monkeypatch.setattr(upload_module, "MegaApi", FakeMegaApi)

    result = CliRunner().invoke(
        app,
        [
            "upload",
            "mega/demo",
            str(source),
            "--token",
            "mega-token",
            "--commit-description",
            "A compact release note.",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "repo_id": "mega/demo",
        "path": "weights.mega",
        "revision": "rev-1",
    }


def test_folder_upload_maps_filters_and_sync_to_native_api(tmp_path, monkeypatch):
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    (source / "skip.tmp").write_bytes(b"skip")

    class FakeCommitInfo:
        oid = "rev-folder"

    class FakeMegaApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, repo_id, **kwargs):
            assert repo_id == "mega/demo"

        def upload_folder(self, **kwargs):
            assert kwargs == {
                "repo_id": "mega/demo",
                "folder_path": source,
                "path_in_repo": "checkpoints",
                "revision": "main",
                "repo_type": "model",
                "commit_message": "sync checkpoint",
                "commit_description": None,
                "allow_patterns": ["*.bin"],
                "ignore_patterns": ["*.tmp"],
                "delete_patterns": "**",
            }
            return FakeCommitInfo()

    monkeypatch.setattr(upload_module, "MegaApi", FakeMegaApi)
    result = CliRunner().invoke(
        app,
        [
            "upload",
            "mega/demo",
            str(source),
            "checkpoints",
            "--include",
            "*.bin",
            "--exclude",
            "*.tmp",
            "--sync",
            "--commit-message",
            "sync checkpoint",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "repo_id": "mega/demo",
        "files": 1,
        "revision": "rev-folder",
    }


def test_folder_upload_forwards_selective_remote_deletions(tmp_path, monkeypatch):
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")

    class FakeCommitInfo:
        oid = "rev-delete"

    class FakeMegaApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, repo_id, **kwargs):
            assert repo_id == "mega/demo"

        def upload_folder(self, **kwargs):
            assert kwargs["delete_patterns"] == ["old/**", "tmp/**"]
            return FakeCommitInfo()

    monkeypatch.setattr(upload_module, "MegaApi", FakeMegaApi)
    result = CliRunner().invoke(
        app,
        ["upload", "mega/demo", str(source), "--delete", "old/**", "--delete", "tmp/**", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["revision"] == "rev-delete"


def test_upload_create_pr_uses_a_real_branch_then_native_discussion(tmp_path, monkeypatch):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"weights")
    calls = []

    class FakeCommitInfo:
        oid = "rev-pr"

    class FakeMegaApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, repo_id, **kwargs):
            calls.append(("repo", repo_id, kwargs))

        def create_branch(self, repo_id, **kwargs):
            calls.append(("branch", repo_id, kwargs))

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs))
            return FakeCommitInfo()

        def create_pull_request(self, repo_id, title, **kwargs):
            calls.append(("pr", repo_id, title, kwargs))
            return type("PullRequest", (), {"url": "https://hub.example.test/mega/demo/discussions/7"})()

    monkeypatch.setattr(upload_module, "MegaApi", FakeMegaApi)
    monkeypatch.setattr(upload_module, "token_hex", lambda _size: "abc123def456")
    result = CliRunner().invoke(
        app,
        [
            "upload", "mega/demo", str(source), "--create-pr", "--commit-message", "Add weights",
            "--commit-description", "Explain the change", "--format", "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("repo", "mega/demo", {"repo_type": "model", "private": False, "exist_ok": True}),
        ("branch", "mega/demo", {"branch": "upload-abc123def456", "revision": "main", "repo_type": "model"}),
        ("upload", {
            "repo_id": "mega/demo", "path_or_fileobj": source, "path_in_repo": "weights.bin",
            "revision": "upload-abc123def456", "commit_message": "Add weights",
            "commit_description": "Explain the change", "repo_type": "model",
        }),
        ("pr", "mega/demo", "Add weights", {
            "description": "Explain the change", "repo_type": "model",
            "source_branch": "upload-abc123def456", "target_branch": "main",
        }),
    ]
    assert json.loads(result.output)["pull_request"] == "https://hub.example.test/mega/demo/discussions/7"


def test_upload_every_uses_the_local_commit_scheduler(tmp_path, monkeypatch):
    source = tmp_path / "watch"
    source.mkdir()
    calls = []

    class FakeScheduler:
        repo_id = "mega/demo"

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(upload_module, "CommitScheduler", FakeScheduler)
    monkeypatch.setattr(upload_module, "MegaApi", lambda **kwargs: ("api", kwargs))
    monkeypatch.setattr(upload_module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    result = CliRunner().invoke(
        app,
        ["upload", "mega/demo", str(source), "--every", "2.5", "--include", "*.jsonl", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [{
        "repo_id": "mega/demo", "folder_path": source, "every": 2.5, "path_in_repo": "",
        "repo_type": "model", "revision": "main", "private": False, "token": None,
        "allow_patterns": ["*.jsonl"], "ignore_patterns": None,
        "mega_api": ("api", {"token": None, "library_name": "mega-cli"}),
    }, "stop"]
    assert json.loads(result.output) == {"repo_id": "mega/demo"}

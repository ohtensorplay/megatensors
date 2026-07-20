from pathlib import Path

import pytest
import click
from click.testing import CliRunner

from megatensors._hub.cli._cli_utils import RepoType
from megatensors._hub.cli.repos import repo_files_cli, repo_files_delete
from megatensors._hub.cli.upload import upload_large_folder


def test_upload_large_folder_delegates_to_mega_upload_client(tmp_path, monkeypatch):
    calls = []

    def fake_run_upload(repo_id, **kwargs):
        calls.append((repo_id, kwargs))

    monkeypatch.setattr("megatensors._hub.cli.upload.run_upload", fake_run_upload)
    source = Path(tmp_path) / "checkpoint"
    source.mkdir()

    upload_large_folder(
        "mega/demo",
        source,
        repo_type=RepoType.dataset,
        revision="release",
        private=True,
        include=["*.jsonl"],
        exclude=["*.tmp"],
        num_workers=3,
        token="mega-token",
    )

    assert calls == [
        (
            "mega/demo",
            {
                "local_path": source,
                "repo_type": "dataset",
                "revision": "release",
                "private": True,
                "include": ["*.jsonl"],
                "exclude": ["*.tmp"],
                "max_workers": 3,
                "sync": False,
                "token": "mega-token",
            },
        )
    ]


def test_upload_large_folder_rejects_files(tmp_path):
    source = Path(tmp_path) / "weights.mega"
    source.write_bytes(b"weights")

    with pytest.raises(click.BadParameter, match="Directory uploads require a local directory"):
        upload_large_folder("mega/demo", source)


def test_repo_files_delete_delegates_to_mega_repos_delete(monkeypatch):
    calls = []

    def fake_repo_delete_files(repo_id, **kwargs):
        calls.append((repo_id, kwargs))

    monkeypatch.setattr("megatensors._hub.cli.repos.repo_delete_files", fake_repo_delete_files)

    repo_files_delete(
        "mega/demo",
        ["*.json", "artifacts/"],
        revision="release",
        commit_message="prune artifacts",
        yes=True,
        token="mega-token",
    )

    assert calls == [
        (
            "mega/demo",
            {
                "patterns": ["*.json", "artifacts/"],
                "revision": "release",
                "commit_message": "prune artifacts",
                "yes": True,
                "token": "mega-token",
            },
        )
    ]


def test_repo_files_help_describes_the_mega_command():
    result = CliRunner().invoke(repo_files_cli, ["--help"], prog_name="mega repo-files")

    assert result.exit_code == 0
    assert "MEGA Hub" in result.output
    assert "mega repo-files delete" in result.output

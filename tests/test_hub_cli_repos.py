from __future__ import annotations

from click.testing import CliRunner

from megatensors._hub.cli import repos
from megatensors.hub import CommitChange, CommitDetail, CommitInfo, RefInfo, RepoInfo, RepoRefs


def test_repos_settings_forwards_only_worker_supported_metadata(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, endpoint=None, token=None):
            calls.append(("init", endpoint, token))

        def update_repo(self, repo_id, **kwargs):
            calls.append(("update_repo", repo_id, kwargs))
            return RepoInfo(
                repo_id=repo_id,
                private=kwargs["private"],
                created_at="created",
                updated_at="updated",
                description=kwargs["description"] or "",
                tags=tuple(kwargs["tags"] or ()),
                license=kwargs["license"] or "",
            )

    monkeypatch.setattr(repos, "MegaHubClient", FakeMegaClient)

    result = CliRunner().invoke(
        repos.repos_cli,
        [
            "settings",
            "mega/demo",
            "--private",
            "--description",
            "New release",
            "--tag",
            "vision",
            "--tag",
            "weights",
            "--license",
            "apache-2.0",
            "--token",
            "mega-token",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[-1] == (
        "update_repo",
        "mega/demo",
        {
            "private": True,
            "description": "New release",
            "tags": ["vision", "weights"],
            "license": "apache-2.0",
        },
    )
    assert CliRunner().invoke(repos.repos_cli, ["update"]).exit_code != 0


def test_repos_move_uses_worker_lifecycle_endpoint(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, endpoint=None, token=None):
            calls.append(("init", endpoint, token))

        def move_repo(self, from_id, to_id):
            calls.append(("move_repo", from_id, to_id))
            return RepoInfo(
                repo_id=to_id,
                private=False,
                created_at="created",
                updated_at="updated",
                owner="mega",
            )

    monkeypatch.setattr(repos, "MegaHubClient", FakeMegaClient)
    result = CliRunner().invoke(repos.repos_cli, ["move", "mega/old", "mega/new"])

    assert result.exit_code == 0, result.output
    assert calls[-1] == ("move_repo", "mega/old", "mega/new")
    assert "repo_id=mega/new" in result.output


def test_repos_duplicate_reuses_server_side_repository_copy(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, endpoint=None, token=None):
            pass

        def duplicate_repo(self, from_id, to_id, **kwargs):
            calls.append((from_id, to_id, kwargs))
            return RepoInfo(
                repo_id=to_id,
                private=bool(kwargs["private"]),
                created_at="created",
                updated_at="updated",
            )

    monkeypatch.setattr(repos, "MegaHubClient", FakeMegaClient)
    result = CliRunner().invoke(
        repos.repos_cli,
        ["duplicate", "mega/source", "mega/copy", "--private"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("mega/source", "mega/copy", {"private": True})]
    assert "repo_id=mega/copy" in result.output


def test_repos_branch_and_tag_commands_reuse_hub_refs(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, endpoint=None, token=None):
            pass

        def list_refs(self, repo_id):
            calls.append(("list_refs", repo_id))
            return RepoRefs(
                branches=(RefInfo("main", "refs/heads/main", "rev-main"),),
                tags=(RefInfo("v1", "refs/tags/v1", "rev-main", message="release"),),
            )

        def create_branch(self, repo_id, branch, **kwargs):
            calls.append(("create_branch", repo_id, branch, kwargs))
            return RefInfo(branch, f"refs/heads/{branch}", "rev-main")

        def create_tag(self, repo_id, tag, **kwargs):
            calls.append(("create_tag", repo_id, tag, kwargs))
            return RefInfo(tag, f"refs/tags/{tag}", "rev-main", message=kwargs.get("message"))

    monkeypatch.setattr(repos, "MegaHubClient", FakeMegaClient)
    runner = CliRunner()
    branch_result = runner.invoke(
        repos.repos_cli,
        ["branch", "create", "mega/demo", "dev", "--revision", "main", "--exist-ok"],
    )
    tag_result = runner.invoke(
        repos.repos_cli,
        ["tag", "create", "mega/demo", "v2", "--message", "release"],
    )

    assert branch_result.exit_code == 0, branch_result.output
    assert tag_result.exit_code == 0, tag_result.output
    assert calls == [
        ("create_branch", "mega/demo", "dev", {"revision": "main", "exist_ok": True}),
        ("create_tag", "mega/demo", "v2", {"revision": "main", "message": "release", "exist_ok": False}),
    ]


def test_repos_history_and_commit_detail_follow_requested_revision(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, endpoint=None, token=None):
            pass

        def list_commits(self, repo_id, **kwargs):
            calls.append(("list_commits", repo_id, kwargs))
            return ([CommitInfo(
                "rev-2",
                "rev-1",
                "release",
                "alice",
                "now",
                signature_status="verified",
                signer_fingerprint="A1B2C3D4",
                signer_subject="alice@example.test",
                author_email="alice@example.test",
            )], None)

        def get_commit(self, repo_id, revision):
            calls.append(("get_commit", repo_id, revision))
            return CommitDetail(
                revision="rev-2",
                parent_revision="rev-1",
                message="release",
                author="alice",
                created_at="now",
                files=(CommitChange("model.mega", "added", 4, "a" * 64),),
                signature_status="verified",
                signer_fingerprint="A1B2C3D4",
                signer_subject="alice@example.test",
                author_email="alice@example.test",
            )

    monkeypatch.setattr(repos, "MegaHubClient", FakeMegaClient)
    runner = CliRunner()
    history_result = runner.invoke(repos.repos_cli, ["history", "mega/demo", "--revision", "release"])
    commit_result = runner.invoke(repos.repos_cli, ["commit", "mega/demo", "release"])

    assert history_result.exit_code == 0, history_result.output
    assert commit_result.exit_code == 0, commit_result.output
    assert calls == [
        ("list_commits", "mega/demo", {"revision": "release", "limit": 50}),
        ("get_commit", "mega/demo", "release"),
    ]
    assert "model.mega" in commit_result.output
    assert "verified" in history_result.output
    assert "A1B2C3D4" in commit_result.output

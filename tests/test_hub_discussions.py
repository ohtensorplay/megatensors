# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from click.testing import CliRunner

from megatensors._hub.cli import discussions
from megatensors.mega_hub import MegaApi
from megatensors.hub import (
    CommitChange,
    CommitDetail,
    CommunityAuthor,
    DiscussionInfo,
    DiscussionMessageInfo,
    DiscussionPage,
    DiscussionPermissions,
    DiscussionThread,
    MegaHubClient,
    PullRequestInfo,
)


def _summary(*, title: str = "Community contract", status: str = "open", pull_request: bool = False) -> dict:
    return {
        "number": 7,
        "title": title,
        "kind": "pull_request" if pull_request else "discussion",
        "status": "closed" if status == "merged" else status,
        "author": {"handle": "alice", "display_name": "Alice", "avatar_url": None},
        "reply_count": 1,
        "reaction_count": 2,
        "created_at": "2026-07-13T00:00:00.000Z",
        "updated_at": "2026-07-13T01:00:00.000Z",
        "pull_request": {
            "source_branch": "feature/community",
            "target_branch": "main",
            "source_revision": "source-revision",
            "target_revision": "target-revision",
            "merged_at": "2026-07-13T02:00:00.000Z" if status == "merged" else None,
            "merged_by": "alice" if status == "merged" else None,
        }
        if pull_request
        else None,
    }


def _thread_payload(*, title: str = "Community contract", status: str = "open") -> dict:
    return {
        "discussion": _summary(title=title, status=status),
        "messages": [
            {
                "message_id": "00000000-0000-4000-8000-000000000001",
                "author": {"handle": "alice", "display_name": "Alice", "avatar_url": None},
                "body": "Original context",
                "is_original": True,
                "created_at": "2026-07-13T00:00:00.000Z",
                "updated_at": "2026-07-13T00:00:00.000Z",
                "reaction": {"type": "fire", "count": 2, "viewer_reacted": True},
                "permissions": {"can_edit": True, "can_delete": False},
            },
            {
                "message_id": "00000000-0000-4000-8000-000000000002",
                "author": {"handle": "bob", "display_name": "Bob", "avatar_url": None},
                "body": "Reply context",
                "is_original": False,
                "created_at": "2026-07-13T01:00:00.000Z",
                "updated_at": "2026-07-13T01:00:00.000Z",
                "reaction": {"type": "fire", "count": 0, "viewer_reacted": False},
                "permissions": {"can_edit": False, "can_delete": False},
            },
        ],
        "permissions": {
            "can_reply": status == "open",
            "can_close": status == "open",
            "can_reopen": status == "closed",
            "can_delete": True,
            "can_merge": False,
            "merge_blocked_reason": None,
        },
    }


def test_native_discussion_client_maps_only_canonical_worker_routes():
    calls = []

    class RecordingClient(MegaHubClient):
        def _request_json(self, method, path, **kwargs):  # type: ignore[override]
            calls.append((method, path, kwargs))
            if method == "GET" and "?" in path:
                return {
                    "discussions": [_summary()],
                    "counts": {"all": 1, "open": 1, "closed": 0, "discussions": 1, "pull_requests": 0},
                    "filter": {"status": "open", "kind": "all", "sort": "recently-updated", "query": "contract"},
                    "page": 2,
                    "limit": 5,
                    "has_more": False,
                }
            if path.endswith("/reactions/fire"):
                return {"reaction": {"type": "fire", "count": 3, "viewer_reacted": True}}
            title = "Renamed contract" if method == "PATCH" else "Community contract"
            return _thread_payload(title=title)

    client = RecordingClient(endpoint="https://hub.example.test", token="secret")
    page = client.list_discussions(
        "mega/demo",
        status="open",
        kind="all",
        sort="recently-updated",
        search="contract",
        page=2,
        limit=5,
    )
    created = client.create_discussion("mega/demo", title="Community contract", body="Durable context")
    renamed = client.update_discussion("mega/demo", 7, title="Renamed contract")
    reaction = client.set_discussion_reaction(
        "mega/demo",
        7,
        "00000000-0000-4000-8000-000000000001",
    )

    assert page.discussions[0].url == "https://hub.example.test/mega/demo/discussions/7"
    assert page.query == "contract"
    assert created.messages[-1].message_id.endswith("2")
    assert renamed.discussion.title == "Renamed contract"
    assert reaction == {"type": "fire", "count": 3, "viewer_reacted": True}
    assert calls == [
        (
            "GET",
            "/api/repos/mega/demo/discussions?status=open&kind=all&sort=recently-updated&page=2&limit=5&q=contract",
            {},
        ),
        (
            "POST",
            "/api/repos/mega/demo/discussions",
            {"json_body": {"title": "Community contract", "body": "Durable context", "kind": "discussion"}, "auth": True},
        ),
        (
            "PATCH",
            "/api/repos/mega/demo/discussions/7",
            {"json_body": {"title": "Renamed contract"}, "auth": True},
        ),
        (
            "PUT",
            "/api/repos/mega/demo/discussions/7/messages/00000000-0000-4000-8000-000000000001/reactions/fire",
            {"auth": True},
        ),
    ]


def _native_thread(*, title: str = "Community contract", status: str = "open") -> DiscussionThread:
    author = CommunityAuthor(handle="alice", display_name="Alice")
    info = DiscussionInfo(
        repo_id="mega/demo",
        repo_type="model",
        number=7,
        title=title,
        kind="discussion",
        status=status,
        author=author,
        reply_count=1,
        reaction_count=2,
        created_at="2026-07-13T00:00:00.000Z",
        updated_at="2026-07-13T01:00:00.000Z",
        endpoint="https://hub.example.test",
    )
    messages = (
        DiscussionMessageInfo(
            message_id="00000000-0000-4000-8000-000000000001",
            author=author,
            body="Original context",
            is_original=True,
            created_at=info.created_at,
            updated_at=info.updated_at,
            reaction_count=2,
            viewer_reacted=True,
            can_edit=True,
            can_delete=False,
        ),
        DiscussionMessageInfo(
            message_id="00000000-0000-4000-8000-000000000002",
            author=author,
            body="Reply context",
            is_original=False,
            created_at=info.updated_at,
            updated_at=info.updated_at,
            reaction_count=0,
            viewer_reacted=False,
            can_edit=True,
            can_delete=True,
        ),
    )
    return DiscussionThread(
        discussion=info,
        messages=messages,
        permissions=DiscussionPermissions(
            can_reply=status == "open",
            can_close=status == "open",
            can_reopen=status == "closed",
            can_delete=True,
            can_merge=False,
        ),
    )


def test_discussions_cli_reuses_hf_interaction_patterns_with_native_client(tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def list_discussions(self, repo_id, **kwargs):
            calls.append(("list", repo_id, kwargs))
            thread = _native_thread()
            return DiscussionPage(
                discussions=(thread.discussion,),
                counts={"all": 1, "open": 1},
                status="open",
                kind="all",
                sort="recently-created",
                query="",
                page=1,
                limit=30,
                has_more=False,
            )

        def create_discussion(self, repo_id, **kwargs):
            calls.append(("create", repo_id, kwargs))
            pull_request = PullRequestInfo("feature/community", "main", "source", "target")
            thread = _native_thread()
            return DiscussionThread(
                discussion=DiscussionInfo(
                    **{
                        **thread.discussion.__dict__,
                        "kind": "pull_request",
                        "pull_request": pull_request,
                    }
                ),
                messages=thread.messages,
                permissions=thread.permissions,
            )

        def reply_to_discussion(self, repo_id, number, body, **kwargs):
            calls.append(("comment", repo_id, number, body, kwargs))
            return _native_thread()

        def update_discussion(self, repo_id, number, **kwargs):
            calls.append(("update", repo_id, number, kwargs))
            return _native_thread(title=kwargs.get("title", "Community contract"), status=kwargs.get("status", "open"))

        def set_discussion_reaction(self, repo_id, number, message_id, **kwargs):
            calls.append(("react", repo_id, number, message_id, kwargs))
            return {"type": "fire", "count": 0, "viewer_reacted": False}

    monkeypatch.setattr(discussions, "MegaHubClient", FakeClient)
    proposal = tmp_path / "proposal.md"
    proposal.write_text("A branch-backed proposal.\n", encoding="utf-8")
    runner = CliRunner()

    listed = runner.invoke(discussions.discussions_cli, ["list", "mega/demo", "--format", "json"])
    created = runner.invoke(
        discussions.discussions_cli,
        [
            "create",
            "mega/demo",
            "--title",
            "Ship community",
            "--body-file",
            str(proposal),
            "--pull-request",
            "--source-branch",
            "feature/community",
            "--format",
            "json",
        ],
    )
    commented = runner.invoke(
        discussions.discussions_cli,
        ["comment", "mega/demo", "7", "--body-file", "-", "--format", "json"],
        input="Context from stdin.\n",
    )
    closed = runner.invoke(
        discussions.discussions_cli,
        ["close", "mega/demo", "7", "--comment", "Resolved.", "--yes", "--format", "json"],
    )
    renamed = runner.invoke(
        discussions.discussions_cli,
        ["rename", "mega/demo", "7", "Clarified title", "--format", "json"],
    )
    reacted = runner.invoke(
        discussions.discussions_cli,
        [
            "react",
            "mega/demo",
            "7",
            "00000000-0000-4000-8000-000000000001",
            "--remove",
            "--format",
            "json",
        ],
    )

    for result in (listed, created, commented, closed, renamed, reacted):
        assert result.exit_code == 0, result.output
        json.loads(result.output)
    assert calls[1] == (
        "list",
        "mega/demo",
        {
            "status": "open",
            "kind": "all",
            "sort": "recently-created",
            "page": 1,
            "limit": 30,
            "repo_type": "model",
        },
    )
    assert ("create", "mega/demo", {
        "title": "Ship community",
        "body": "A branch-backed proposal.\n",
        "kind": "pull_request",
        "source_branch": "feature/community",
        "target_branch": "main",
        "repo_type": "model",
    }) in calls
    assert ("comment", "mega/demo", 7, "Context from stdin.\n", {"repo_type": "model"}) in calls
    assert ("update", "mega/demo", 7, {"status": "closed", "comment": "Resolved.", "repo_type": "model"}) in calls
    assert ("update", "mega/demo", 7, {"title": "Clarified title", "repo_type": "model"}) in calls
    assert (
        "react",
        "mega/demo",
        7,
        "00000000-0000-4000-8000-000000000001",
        {"active": False, "repo_type": "model"},
    ) in calls


def test_hf_compatible_python_api_delegates_to_native_community_client(monkeypatch):
    calls = []

    class FakeService:
        def list_discussions(self, repo_id, **kwargs):
            calls.append(("list", repo_id, kwargs))
            thread = _native_thread()
            return DiscussionPage(
                discussions=(thread.discussion,),
                counts={"all": 1, "open": 1},
                status="all",
                kind="all",
                sort="recently-created",
                query="",
                page=1,
                limit=50,
                has_more=False,
            )

        def get_discussion(self, repo_id, number, **kwargs):
            calls.append(("get", repo_id, number, kwargs))
            return _native_thread()

        def create_discussion(self, repo_id, **kwargs):
            calls.append(("create", repo_id, kwargs))
            return _native_thread()

        def reply_to_discussion(self, repo_id, number, body, **kwargs):
            calls.append(("reply", repo_id, number, body, kwargs))
            return _native_thread()

        def update_discussion(self, repo_id, number, **kwargs):
            calls.append(("update", repo_id, number, kwargs))
            return _native_thread(title=kwargs.get("title", "Community contract"), status=kwargs.get("status", "open"))

    service = FakeService()
    monkeypatch.setattr("megatensors._hub.mega_api._mega_service_client", lambda api, token=None: service)
    api = MegaApi(endpoint="https://hub.example.test", token="secret")

    listed = list(api.get_repo_discussions("mega/demo", repo_type="model"))
    detail = api.get_discussion_details("mega/demo", 7, repo_type="model")
    created = api.create_discussion("mega/demo", "Community contract", description="Context", repo_type="model")
    comment = api.comment_discussion("mega/demo", 7, "Reply context", repo_type="model")
    renamed = api.rename_discussion("mega/demo", 7, "Clarified title", repo_type="model")
    closed = api.change_discussion_status("mega/demo", 7, "closed", comment="Resolved", repo_type="model")

    assert listed[0].num == 7 and listed[0].author == "alice"
    assert detail.events[-1].id.endswith("2")
    assert created.title == "Community contract"
    assert comment.content == "Reply context"
    assert renamed.new_title == "Clarified title"
    assert closed.new_status == "closed"
    assert calls[0] == (
        "list",
        "mega/demo",
        {"status": "all", "kind": "all", "page": 1, "limit": 50, "repo_type": "model"},
    )


def test_discussions_diff_reuses_native_commit_graph(monkeypatch):
    base = _native_thread()
    pull_request = PullRequestInfo(
        source_branch="feature/community",
        target_branch="main",
        source_revision="source-revision",
        target_revision="target-revision",
    )
    thread = DiscussionThread(
        discussion=DiscussionInfo(
            **{
                **base.discussion.__dict__,
                "kind": "pull_request",
                "pull_request": pull_request,
            }
        ),
        messages=base.messages,
        permissions=base.permissions,
    )
    calls = []

    class FakeClient:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def get_discussion(self, repo_id, number, **kwargs):
            calls.append(("get", repo_id, number, kwargs))
            return thread

        def get_commit(self, repo_id, revision):
            calls.append(("commit", repo_id, revision))
            return CommitDetail(
                revision="source-revision",
                parent_revision="target-revision",
                message="Ship community",
                author="alice",
                created_at="2026-07-13T02:00:00.000Z",
                files=(
                    CommitChange(
                        path="README.md",
                        change="modified",
                        size=42,
                        sha256="a" * 64,
                        previous_sha256="b" * 64,
                    ),
                ),
            )

    monkeypatch.setattr(discussions, "MegaHubClient", FakeClient)
    result = CliRunner().invoke(
        discussions.discussions_cli,
        ["diff", "mega/demo", "7", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["base_revision"] == "target-revision"
    assert payload["head_revision"] == "source-revision"
    assert payload["commit_count"] == 1
    assert payload["changes"][0]["path"] == "README.md"
    assert calls[-1] == ("commit", "mega/demo", "source-revision")

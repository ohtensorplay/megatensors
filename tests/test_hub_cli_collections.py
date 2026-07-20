import json

from click.testing import CliRunner

from megatensors._hub.cli import collections
from megatensors.hub import CollectionInfo, CollectionItemInfo, CollectionOwnerInfo


def collection_info() -> CollectionInfo:
    return CollectionInfo(
        id="collection_01",
        slug="mega/portable-models",
        title="Portable models",
        description="Models that run close to the user.",
        private=False,
        theme="green",
        position=0,
        owner=CollectionOwnerInfo(
            handle="mega",
            display_name="MEGA",
            kind="organization",
        ),
        item_count=1,
        upvotes=42,
        upvoted=True,
        items=(
            CollectionItemInfo(
                item_object_id="item_01",
                item_id="mega/mega-3b-base",
                item_type="model",
                title="Mega 3B Base",
                description="A compact base model.",
                href="/mega/mega-3b-base",
                note="Runs locally.",
                position=0,
                created_at="2026-07-16T00:00:00Z",
                updated_at="2026-07-16T00:00:00Z",
            ),
        ),
        created_at="2026-07-16T00:00:00Z",
        updated_at="2026-07-17T00:00:00Z",
        url="https://mega.tensorplay.cn/collections/mega/portable-models",
    )


def test_collections_list_forwards_hf_compatible_filters(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, token=None):
            calls.append(("client", token))

        def list_collections(self, **kwargs):
            calls.append(("list", kwargs))
            return [collection_info()]

    monkeypatch.setattr(collections, "MegaHubClient", FakeMegaClient)

    result = CliRunner().invoke(
        collections.collections_cli,
        [
            "list",
            "--owner",
            "mega",
            "--item",
            "models/mega/mega-3b-base",
            "--sort",
            "upvotes",
            "--search",
            "portable",
            "--limit",
            "7",
            "--token",
            "mega_test",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {
            "slug": "mega/portable-models",
            "title": "Portable models",
            "items": 1,
            "upvotes": 42,
            "visibility": "public",
            "theme": "green",
            "updated_at": "2026-07-17T00:00:00Z",
        }
    ]
    assert calls == [
        ("client", "mega_test"),
        (
            "list",
            {
                "owner": "mega",
                "item": "models/mega/mega-3b-base",
                "sort": "upvotes",
                "search": "portable",
                "limit": 7,
            },
        ),
    ]


def test_collections_metadata_commands_forward_all_fields(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, token=None):
            calls.append(("client", token))

        def create_collection(self, title, **kwargs):
            calls.append(("create", title, kwargs))
            return collection_info()

        def update_collection(self, slug, **kwargs):
            calls.append(("update", slug, kwargs))
            return collection_info()

        def delete_collection(self, slug, **kwargs):
            calls.append(("delete", slug, kwargs))

    monkeypatch.setattr(collections, "MegaHubClient", FakeMegaClient)
    runner = CliRunner()

    created = runner.invoke(
        collections.collections_cli,
        [
            "create",
            "Portable models",
            "--namespace",
            "mega",
            "--description",
            "Close to the user",
            "--private",
            "--theme",
            "green",
            "--exists-ok",
            "--format",
            "json",
        ],
    )
    updated = runner.invoke(
        collections.collections_cli,
        [
            "update",
            "mega/portable-models",
            "--title",
            "Portable AI",
            "--description",
            "Updated",
            "--position",
            "2",
            "--public",
            "--theme",
            "blue",
            "--format",
            "json",
        ],
    )
    deleted = runner.invoke(
        collections.collections_cli,
        [
            "delete",
            "mega/portable-models",
            "--missing-ok",
            "--format",
            "json",
        ],
    )

    assert created.exit_code == 0, created.output
    assert updated.exit_code == 0, updated.output
    assert deleted.exit_code == 0, deleted.output
    assert json.loads(created.output)["slug"] == "mega/portable-models"
    assert json.loads(updated.output)["slug"] == "mega/portable-models"
    assert json.loads(deleted.output) == {"slug": "mega/portable-models"}
    assert calls == [
        ("client", None),
        (
            "create",
            "Portable models",
            {
                "namespace": "mega",
                "description": "Close to the user",
                "private": True,
                "theme": "green",
                "exists_ok": True,
            },
        ),
        ("client", None),
        (
            "update",
            "mega/portable-models",
            {
                "title": "Portable AI",
                "description": "Updated",
                "position": 2,
                "private": False,
                "theme": "blue",
            },
        ),
        ("client", None),
        ("delete", "mega/portable-models", {"missing_ok": True}),
    ]


def test_collection_item_commands_preserve_stable_item_ids(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, token=None):
            pass

        def add_collection_item(self, slug, item_id, item_type, **kwargs):
            calls.append(("add", slug, item_id, item_type, kwargs))
            return collection_info()

        def update_collection_item(self, slug, item_object_id, **kwargs):
            calls.append(("update", slug, item_object_id, kwargs))
            return collection_info()

        def delete_collection_item(self, slug, item_object_id, **kwargs):
            calls.append(("delete", slug, item_object_id, kwargs))
            return collection_info()

    monkeypatch.setattr(collections, "MegaHubClient", FakeMegaClient)
    runner = CliRunner()

    added = runner.invoke(
        collections.collections_cli,
        [
            "add-item",
            "mega/portable-models",
            "mega/mega-3b-base",
            "model",
            "--note",
            "Runs locally",
            "--position",
            "1",
            "--exists-ok",
            "--format",
            "json",
        ],
    )
    updated = runner.invoke(
        collections.collections_cli,
        [
            "update-item",
            "mega/portable-models",
            "item_01",
            "--note",
            "Validated",
            "--position",
            "0",
            "--format",
            "json",
        ],
    )
    deleted = runner.invoke(
        collections.collections_cli,
        [
            "delete-item",
            "mega/portable-models",
            "item_01",
            "--missing-ok",
            "--format",
            "json",
        ],
    )

    assert added.exit_code == 0, added.output
    assert updated.exit_code == 0, updated.output
    assert deleted.exit_code == 0, deleted.output
    assert calls == [
        (
            "add",
            "mega/portable-models",
            "mega/mega-3b-base",
            "model",
            {"note": "Runs locally", "position": 1, "exists_ok": True},
        ),
        (
            "update",
            "mega/portable-models",
            "item_01",
            {"note": "Validated", "position": 0},
        ),
        (
            "delete",
            "mega/portable-models",
            "item_01",
            {"missing_ok": True},
        ),
    ]


def test_collection_item_commands_accept_storage_buckets(monkeypatch):
    calls = []

    class FakeMegaClient:
        def __init__(self, *, token=None):
            pass

        def add_collection_item(self, slug, item_id, item_type, **kwargs):
            calls.append((slug, item_id, item_type, kwargs))
            return collection_info()

    monkeypatch.setattr(collections, "MegaHubClient", FakeMegaClient)
    result = CliRunner().invoke(
        collections.collections_cli,
        [
            "add-item",
            "mega/portable-models",
            "mega/release-artifacts",
            "bucket",
            "--note",
            "Mutable artifacts",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "mega/portable-models",
            "mega/release-artifacts",
            "bucket",
            {"note": "Mutable artifacts", "position": None, "exists_ok": False},
        )
    ]


def test_collection_item_commands_reject_non_hf_article_type():
    result = CliRunner().invoke(
        collections.collections_cli,
        ["add-item", "mega/portable-models", "release-notes", "article"],
    )

    assert result.exit_code != 0
    assert "article" in result.output

import json

from click.testing import CliRunner

from megatensors._hub._buckets import BucketFile, BucketFolder, BucketInfo, BucketUrl
from megatensors._hub.cli import buckets


def test_buckets_create_uses_mega_uri_and_hf_compatible_options(monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, *, token=None):
            calls.append(("init", token))

        def create_bucket(self, bucket_id, **kwargs):
            calls.append(("create", bucket_id, kwargs))
            return BucketUrl("https://mega.tensorplay.cn/buckets/acme/checkpoints", endpoint="https://mega.tensorplay.cn")

    monkeypatch.setattr(buckets, "MegaApi", FakeApi)
    result = CliRunner().invoke(
        buckets.buckets_cli,
        ["create", "mega://buckets/acme/checkpoints", "--private", "--region", "eu", "--exist-ok", "--token", "secret"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("init", "secret"),
        ("create", "acme/checkpoints", {"private": True, "region": "eu", "exist_ok": True}),
    ]
    assert "Bucket created" in result.output


def test_buckets_list_supports_recursive_tree_output(monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, *, token=None):
            pass

        def list_bucket_tree(self, bucket_id, *, prefix=None, recursive=None):
            calls.append((bucket_id, prefix, recursive))
            return [
                BucketFolder(type="directory", path="runs", uploadedAt="2026-07-19T00:00:00Z"),
                BucketFile(type="file", path="runs/metrics.json", size=1200, xetHash="a" * 64, mtime="2026-07-19T00:00:00Z", uploadedAt="2026-07-19T00:00:00Z"),
            ]

    monkeypatch.setattr(buckets, "MegaApi", FakeApi)
    result = CliRunner().invoke(
        buckets.buckets_cli,
        ["list", "acme/checkpoints", "--tree", "--recursive", "--human-readable", "--format", "human"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("acme/checkpoints", None, True)]
    assert "runs/" in result.output
    assert "metrics.json" in result.output
    assert "KB" in result.output


def test_buckets_list_preserves_hf_json_field_names(monkeypatch):
    class FakeApi:
        def __init__(self, *, token=None):
            pass

        def list_buckets(self, *, namespace=None, search=None):
            return [BucketInfo(
                id="acme/checkpoints",
                private=True,
                createdAt="2026-07-19T00:00:00.000Z",
                size=42,
                totalFiles=3,
            )]

    monkeypatch.setattr(buckets, "MegaApi", FakeApi)
    result = CliRunner().invoke(
        buckets.buckets_cli,
        ["list", "acme", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    item = json.loads(result.output)[0]
    assert item == {
        "id": "acme/checkpoints",
        "private": True,
        "size": 42,
        "total_files": 3,
        "created_at": "2026-07-19 00:00:00+00:00",
    }


def test_buckets_tree_is_rejected_for_namespace_listing():
    result = CliRunner().invoke(buckets.buckets_cli, ["list", "acme", "--tree"])
    assert result.exit_code != 0
    assert "--tree requires a Bucket ID" in result.output

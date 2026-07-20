from __future__ import annotations

from pathlib import Path

from megatensors._hub._commit_api import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)
from megatensors._hub.mega_api import MegaApi, _mega_create_commit


def test_mega_create_commit_routes_additions_and_deletions_through_native_xet(
    tmp_path, monkeypatch
):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"weights")
    addition = CommitOperationAdd(path_or_fileobj=source, path_in_repo="weights.bin")
    deletion = CommitOperationDelete(path_in_repo="old.bin")
    expected = object()
    seen = {}

    def fake_pipelined_upload(api, **kwargs):
        seen["api"] = api
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(
        "megatensors._hub._upload_pipeline.pipelined_upload", fake_pipelined_upload
    )
    api = MegaApi(endpoint="https://hub.example", token="token")

    result = _mega_create_commit(
        api,
        "mega/demo",
        [addition, deletion],
        commit_message="native upload",
        commit_description="",
        token=None,
        repo_type="model",
        revision="main",
        create_pr=False,
        parent_commit=None,
    )

    assert result is expected
    assert seen == {
        "api": api,
        "repo_id": "mega/demo",
        "repo_type": "model",
        "add_operations": [addition],
        "delete_operations": [deletion],
        "commit_message": "native upload",
        "commit_description": "",
        "token": None,
        "revision": "main",
        "create_pr": False,
        "parent_commit": None,
    }


def test_mega_create_commit_keeps_copy_operations_off_xet_pipeline(
    tmp_path, monkeypatch
):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"weights")
    operations = [
        CommitOperationAdd(path_or_fileobj=source, path_in_repo="weights.bin"),
        CommitOperationCopy(src_path_in_repo="base.bin", path_in_repo="copy.bin"),
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "copy operations are not supported by the native Xet upload pipeline"
        )

    monkeypatch.setattr(
        "megatensors._hub._upload_pipeline.pipelined_upload", fail_if_called
    )

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def list_refs(self, repo_id):
            class Refs:
                branches = [
                    type("Ref", (), {"name": "main", "target_revision": "parent"})()
                ]

            return Refs()

        def _stage_file(self, *args, **kwargs):
            return {
                "operation": "add",
                "path": "weights.bin",
                "size": 7,
                "sha256": "digest",
            }

        def create_commit(self, *args, **kwargs):
            seen["create_commit"] = (args, kwargs)
            return {"revision": "next"}

    monkeypatch.setattr("megatensors.hub.MegaHubClient", FakeClient)
    api = MegaApi(endpoint="https://hub.example", token="token")

    result = _mega_create_commit(
        api,
        "mega/demo",
        operations,
        commit_message="copy",
        commit_description="Copy operation details",
        token=None,
        repo_type="model",
        revision="main",
        create_pr=False,
        parent_commit=None,
    )

    assert result.oid == "next"
    assert seen["create_commit"][1]["commit_description"] == "Copy operation details"

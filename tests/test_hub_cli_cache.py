from __future__ import annotations

import hashlib

from click.testing import CliRunner

from megatensors._hub.cli import cache
from megatensors._hub.errors import CLIError
from megatensors.hub import FileInfo


def test_cache_rm_accepts_only_mega_repository_uris():
    assert cache._repo_cache_id_from_target("mega://datasets/mega/demo") == "dataset/mega/demo"
    assert cache._repo_cache_id_from_target("mega://mega/demo") == "model/mega/demo"

    try:
        cache._repo_cache_id_from_target("hf://models/mega/demo")
    except CLIError as error:
        assert "mega://" in str(error)
    else:
        raise AssertionError("legacy URI unexpectedly accepted")


def test_cache_verify_uses_worker_file_checksums(tmp_path, monkeypatch):
    payload = b"verified by mega"
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)

    class FakeMegaClient:
        def __init__(self, *, token=None):
            assert token == "mega-token"

        def list_files(self, repo_id, *, revision):
            assert repo_id == "mega/demo"
            assert revision == "main"
            return [FileInfo(path="weights.bin", size=len(payload), sha256=hashlib.sha256(payload).hexdigest())]

    monkeypatch.setattr(cache, "MegaHubClient", FakeMegaClient)

    result = CliRunner().invoke(
        cache.cache_cli,
        [
            "verify",
            "mega/demo",
            "--local-dir",
            str(tmp_path),
            "--token",
            "mega-token",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Verified 1 file(s)" in result.output

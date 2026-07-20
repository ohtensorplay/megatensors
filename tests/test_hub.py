# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from megatensors.cli import app
from megatensors.hub import AccountKeyInfo, MegaHubClient, MegaHubError, RefInfo, RepoInfo, RepoRefs, clear_auth, load_auth, save_auth
from megatensors.mega_hub import MegaApi
from megatensors.mega_hub import MegaFileSystem, mega_hub_url
from megatensors._hub.cli._output import _MEGA_ASCII, out
from megatensors._hub.utils import build_mega_headers


def _configure_token_paths(tmp_path, monkeypatch):
    from megatensors._hub import constants

    monkeypatch.delenv("MEGA_TOKEN", raising=False)
    token_path = tmp_path / "token"
    monkeypatch.setattr(constants, "MEGA_TOKEN_PATH", str(token_path))
    monkeypatch.setattr(constants, "MEGA_STORED_TOKENS_PATH", str(tmp_path / "stored_tokens"))
    monkeypatch.setattr(constants, "ENDPOINT", "https://mega.tensorplay.cn")
    return token_path


def test_auth_token_file_roundtrip_does_not_write_legacy_config(tmp_path, monkeypatch):
    token_path = _configure_token_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("MEGA_HOME", str(tmp_path))
    cfg = save_auth(token="secret-token", endpoint="https://hub.example.test/")

    assert cfg.endpoint == "https://hub.example.test"
    assert cfg.token == "secret-token"
    assert load_auth() == cfg
    assert token_path.read_text(encoding="utf-8") == "secret-token"
    assert not (tmp_path / "config.json").exists()

    clear_auth()
    assert not token_path.exists()
    assert not (tmp_path / "config.json").exists()
    assert load_auth().token is None
    assert load_auth().endpoint == "https://hub.example.test"


def test_cli_login_uses_hub_validation_and_token_storage(tmp_path, monkeypatch):
    token_path = _configure_token_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("MEGA_HOME", str(tmp_path))
    monkeypatch.setattr("megatensors._hub.constants.ENDPOINT", "https://hub.example.test")

    class RecordingApi:
        def __init__(self, endpoint=None):
            assert endpoint == "https://hub.example.test"

        def whoami(self, token):
            assert token == "secret-token"
            return {
                "name": "alice",
                "auth": {"accessToken": {"role": "write", "displayName": "workstation"}},
            }

    monkeypatch.setattr("megatensors._hub.mega_api.MegaApi", RecordingApi)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["auth", "login", "--token", "secret-token"],
    )

    assert result.exit_code == 0
    assert load_auth().endpoint == "https://hub.example.test"
    assert load_auth().token == "secret-token"
    assert token_path.read_text(encoding="utf-8") == "secret-token"
    assert not (tmp_path / "config.json").exists()


def test_cli_login_delegates_browser_flow_to_hub(tmp_path, monkeypatch):
    _configure_token_paths(tmp_path, monkeypatch)
    calls = []

    def fake_login(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("megatensors._hub.cli.auth.login", fake_login)

    out.reset_banner()
    try:
        result = CliRunner().invoke(app, ["auth", "login", "--format", "human"])
    finally:
        out.reset_banner()

    assert result.exit_code == 0
    assert result.stdout == ""
    assert _MEGA_ASCII in result.stderr
    assert "Endpoint:" not in result.stderr
    assert calls == [{"token": None, "add_to_git_credential": False, "skip_if_logged_in": True}]


def test_upload_folder_filters_and_remote_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("MEGA_HOME", str(tmp_path / "home"))
    root = tmp_path / "model"
    (root / "nested").mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "nested" / "model.mega").write_bytes(b"mega")
    (root / "notes.txt").write_text("skip", encoding="utf-8")

    staged: list[tuple[Path, str]] = []
    commits = []

    class RecordingApi(MegaHubClient):
        def create_repo(self, repo_id, **kwargs):  # type: ignore[override]
            return RepoInfo(repo_id=repo_id, private=False, created_at="now", updated_at="now")

        def list_refs(self, repo_id):  # type: ignore[override]
            return RepoRefs(
                branches=(RefInfo("main", "refs/heads/main", "parent-revision"),),
                tags=(),
            )

        def _stage_file(self, repo_id, local_path, **kwargs):  # type: ignore[override]
            remote_path = kwargs["remote_path"]
            staged.append((Path(local_path), remote_path))
            return {
                "operation": "add",
                "path": remote_path,
                "size": Path(local_path).stat().st_size,
                "sha256": "a" * 64,
                "content_type": "application/octet-stream",
            }

        def create_commit(self, repo_id, operations, **kwargs):  # type: ignore[override]
            commits.append((repo_id, list(operations), kwargs))
            return {"revision": "atomic-revision"}

    result = RecordingApi(endpoint="https://hub.example.test", token="token").upload_folder(
        "mega/demo",
        root,
        path_in_repo="weights",
        include=["*.json", "**/*.mega"],
        exclude=["notes.*"],
        max_workers=1,
    )

    assert len(result) == 2
    assert staged == [
        (root / "config.json", "weights/config.json"),
        (root / "nested" / "model.mega", "weights/nested/model.mega"),
    ]
    assert commits[0][2]["parent_revision"] == "parent-revision"
    assert {item["revision"] for item in result} == {"atomic-revision"}


def test_create_commit_sends_atomic_operations_and_parent_guard():
    calls = []

    class RecordingApi(MegaHubClient):
        def _request_json(self, method, path, **kwargs):  # type: ignore[override]
            calls.append((method, path, kwargs))
            return {"revision": "next-revision"}

    result = RecordingApi(endpoint="https://hub.example.test", token="token").create_commit(
        "mega/demo",
        [
            {
                "operation": "add",
                "path": "weights/model.mega",
                "size": 4,
                "sha256": "a" * 64,
                "content_type": "application/octet-stream",
            },
            {"operation": "delete", "path": "old.bin"},
        ],
        revision="main",
        parent_revision="parent-revision",
        commit_message="Atomic update",
        commit_description="Explain the atomic update.",
    )

    assert result["revision"] == "next-revision"
    assert calls == [(
        "POST",
        "/api/repos/mega/demo/commits",
        {
            "json_body": {
                "revision": "main",
                "commit_message": "Atomic update",
                "operations": [
                    {
                        "operation": "add",
                        "path": "weights/model.mega",
                        "size": 4,
                        "sha256": "a" * 64,
                        "content_type": "application/octet-stream",
                    },
                    {"operation": "delete", "path": "old.bin"},
                ],
                "parent_revision": "parent-revision",
                "commit_description": "Explain the atomic update.",
            },
            "auth": True,
        },
    )]


def test_upload_file_with_description_stages_then_commits_atomically(tmp_path):
    source = tmp_path / "config.json"
    source.write_text("{}", encoding="utf-8")
    calls = []

    class RecordingApi(MegaHubClient):
        def create_repo(self, repo_id, **kwargs):  # type: ignore[override]
            calls.append(("repo", repo_id, kwargs))
            return RepoInfo(repo_id=repo_id, private=False, created_at="now", updated_at="now")

        def _stage_file(self, repo_id, local_path, **kwargs):  # type: ignore[override]
            calls.append(("stage", repo_id, Path(local_path), kwargs))
            return {
                "operation": "add",
                "path": kwargs["remote_path"],
                "size": 2,
                "sha256": "a" * 64,
                "content_type": "application/json",
            }

        def create_commit(self, repo_id, operations, **kwargs):  # type: ignore[override]
            calls.append(("commit", repo_id, list(operations), kwargs))
            return {"revision": "described-revision"}

    result = RecordingApi(endpoint="https://hub.example.test", token="token").upload_file(
        "mega/demo",
        source,
        commit_description="Configuration release notes",
    )

    assert result == {
        "repo_id": "mega/demo",
        "path": "config.json",
        "revision": "described-revision",
        "size": 2,
        "sha256": "a" * 64,
    }
    assert calls[-1] == (
        "commit",
        "mega/demo",
        [{
            "operation": "add",
            "path": "config.json",
            "size": 2,
            "sha256": "a" * 64,
            "content_type": "application/json",
        }],
        {
            "revision": "main",
            "commit_message": "Upload config.json",
            "commit_description": "Configuration release notes",
        },
    )


def test_account_key_client_uses_the_native_account_routes_and_rejects_private_material():
    calls = []

    class RecordingApi(MegaHubClient):
        def _request_json(self, method, path, **kwargs):  # type: ignore[override]
            calls.append((method, path, kwargs))
            if method == "GET":
                return {
                    "keys": [
                        {
                            "key_id": "key-1",
                            "key_type": "ssh",
                            "name": "Laptop",
                            "public_key": "ssh-ed25519 AAAA laptop",
                            "fingerprint": "SHA256:abc",
                            "created_at": "2026-07-12T00:00:00.000Z",
                            "revoked_at": None,
                        }
                    ]
                }
            return {
                "key_id": "key-2",
                "key_type": kwargs["json_body"]["key_type"],
                "name": kwargs["json_body"]["name"],
                "public_key": kwargs["json_body"]["public_key"],
                "fingerprint": "A" * 40,
                "created_at": "2026-07-12T00:00:00.000Z",
                "revoked_at": None,
            }

        def _request(self, method, path, **kwargs):  # type: ignore[override]
            calls.append((method, path, kwargs))
            return b""

    client = RecordingApi(endpoint="https://hub.example.test", token="token")
    assert client.list_account_keys() == [
        AccountKeyInfo(
            key_id="key-1",
            key_type="ssh",
            name="Laptop",
            public_key="ssh-ed25519 AAAA laptop",
            fingerprint="SHA256:abc",
            created_at="2026-07-12T00:00:00.000Z",
        )
    ]
    added = client.add_account_key(
        key_type="gpg",
        name="Signing key",
        public_key="-----BEGIN PGP PUBLIC KEY BLOCK-----\npublic\n-----END PGP PUBLIC KEY BLOCK-----",
    )
    assert added.key_id == "key-2"
    client.delete_account_key("key/2")
    assert calls[-1] == ("DELETE", "/api/me/keys/key%2F2", {"auth": True})

    with pytest.raises(ValueError, match="Private keys are never accepted"):
        client.add_account_key(
            key_type="gpg",
            name="Never upload this",
            public_key="-----BEGIN PGP PRIVATE KEY BLOCK-----\nsecret\n-----END PGP PRIVATE KEY BLOCK-----",
        )
    assert len(calls) == 3


def test_hub_api_consumes_repo_and_file_pagination():
    paths = []

    class RecordingApi(MegaHubClient):
        def _request_json(self, method, path, **kwargs):  # type: ignore[override]
            paths.append(path)
            if path.startswith("/api/repos?"):
                if "cursor=" in path:
                    return {
                        "repos": [{"repo_id": "mega/two", "private": False}],
                        "next_cursor": None,
                    }
                return {
                    "repos": [{"repo_id": "mega/one", "private": False}],
                    "next_cursor": "repo-page-2",
                }
            if "cursor=file-page-2" in path:
                return {
                    "revision": "resolved-revision",
                    "files": [{"path": "b.txt", "size": 2, "sha256": "b" * 64}],
                    "next_cursor": None,
                }
            return {
                "revision": "resolved-revision",
                "files": [{"path": "a.txt", "size": 1, "sha256": "a" * 64}],
                "next_cursor": "file-page-2",
            }

    api = RecordingApi(endpoint="https://hub.example.test", token="token")
    repos_found = api.list_repos(limit=2, repo_type="model")
    files_found = api.list_files("mega/one", revision="main")

    assert [repo.repo_id for repo in repos_found] == ["mega/one", "mega/two"]
    assert [file.path for file in files_found] == ["a.txt", "b.txt"]
    assert "cursor=repo-page-2" in paths[1]
    assert "revision=resolved-revision" in paths[3]
    assert "cursor=file-page-2" in paths[3]


def test_upload_folder_sync_adds_and_deletes_in_one_commit(tmp_path):
    root = tmp_path / "folder"
    root.mkdir()
    (root / "keep.txt").write_text("new", encoding="utf-8")
    commits = []

    class RecordingApi(MegaHubClient):
        def create_repo(self, repo_id, **kwargs):  # type: ignore[override]
            return RepoInfo(repo_id=repo_id, private=False, created_at="now", updated_at="now")

        def list_refs(self, repo_id):  # type: ignore[override]
            return RepoRefs(
                branches=(RefInfo("main", "refs/heads/main", "parent-revision"),),
                tags=(),
            )

        def list_files(self, repo_id, *, revision="main"):  # type: ignore[override]
            from megatensors.hub import FileInfo

            return [
                FileInfo("mirror/keep.txt", 3, "a" * 64),
                FileInfo("mirror/remove.txt", 3, "b" * 64),
                FileInfo("outside.txt", 3, "c" * 64),
            ]

        def _stage_file(self, repo_id, local_path, **kwargs):  # type: ignore[override]
            return {
                "operation": "add",
                "path": kwargs["remote_path"],
                "size": 3,
                "sha256": "d" * 64,
                "content_type": "text/plain",
            }

        def create_commit(self, repo_id, operations, **kwargs):  # type: ignore[override]
            commits.append((list(operations), kwargs))
            return {"revision": "sync-revision"}

    result = RecordingApi(endpoint="https://hub.example.test", token="token").upload_folder(
        "mega/demo",
        root,
        path_in_repo="mirror",
        delete_missing=True,
        max_workers=1,
    )

    assert result[0]["revision"] == "sync-revision"
    assert commits == [(
        [
            {
                "operation": "add",
                "path": "mirror/keep.txt",
                "size": 3,
                "sha256": "d" * 64,
                "content_type": "text/plain",
            },
            {"operation": "delete", "path": "mirror/remove.txt"},
        ],
        {
            "revision": "main",
            "parent_revision": "parent-revision",
            "commit_message": "Sync 1 files",
        },
    )]


def test_cli_repo_create_passes_metadata(monkeypatch):
    calls = []

    class RecordingApi:
        def __init__(self, endpoint=None, token=None):
            calls.append(("init", endpoint, token))

        def create_repo(self, repo_id, **kwargs):
            calls.append(("create_repo", repo_id, kwargs))
            from megatensors.hub import RepoInfo

            return RepoInfo(
                repo_id=repo_id,
                private=kwargs["private"],
                created_at="now",
                updated_at="now",
                repo_type=kwargs["repo_type"],
            )

    monkeypatch.setattr("megatensors._hub.cli.repos.MegaHubClient", RecordingApi)
    result = CliRunner().invoke(
        app,
        [
            "repos",
            "create",
            "mega/demo",
            "--repo-type",
            "dataset",
            "--private",
            "--description",
            "Demo dataset",
            "--tag",
            "vision",
            "--license",
            "apache-2.0",
                "--exist-ok",
                "--format",
                "human",
            ],
    )

    assert result.exit_code == 0
    assert calls[-1] == (
        "create_repo",
        "mega/demo",
        {
            "repo_type": "dataset",
            "private": True,
            "description": "Demo dataset",
            "tags": ["vision"],
            "license": "apache-2.0",
            "exist_ok": True,
        },
    )
    assert "Repository created" in result.output


def test_cli_repo_delete_files_expands_patterns(monkeypatch):
    calls = []

    class RecordingApi:
        def __init__(self, endpoint=None, token=None):
            pass

        def list_files(self, repo_id, *, revision="main"):
            from megatensors.hub import FileInfo

            return [
                FileInfo(path="config.json", size=2, sha256="a"),
                FileInfo(path="weights/model.mega", size=4, sha256="b"),
                FileInfo(path="notes.txt", size=1, sha256="c"),
            ]

        def list_refs(self, repo_id):
            from megatensors.hub import RefInfo, RepoRefs

            return RepoRefs(
                branches=(RefInfo(name="main", ref="refs/heads/main", target_revision="parent-rev"),),
                tags=(),
            )

        def create_commit(self, repo_id, operations, **kwargs):
            calls.append((repo_id, list(operations), kwargs))
            return {"revision": "delete-rev"}

    monkeypatch.setattr("megatensors._hub.cli.repos.MegaHubClient", RecordingApi)
    result = CliRunner().invoke(app, [
        "repos", "delete-files", "mega/demo", "*.json", "weights/", "--yes", "--format", "human",
    ])

    assert result.exit_code == 0
    assert calls == [(
        "mega/demo",
        [
            {"operation": "delete", "path": "config.json"},
            {"operation": "delete", "path": "weights/model.mega"},
        ],
        {
            "revision": "main",
            "parent_revision": "parent-rev",
            "commit_message": "Delete 2 files",
        },
    )]
    assert "Files deleted" in result.output


def test_cli_dataset_upload_uses_dataset_repo_type(tmp_path, monkeypatch):
    path = tmp_path / "data.jsonl"
    path.write_text("{}", encoding="utf-8")
    calls = []

    def fake_run_upload(repo_id, **kwargs):
        calls.append((repo_id, kwargs))

    monkeypatch.setattr("megatensors._hub.cli.models.run_upload", fake_run_upload)
    result = CliRunner().invoke(app, ["datasets", "upload", "mega/data", str(path), "--format", "human"])

    assert result.exit_code == 0
    assert calls[0][0] == "mega/data"
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[0][1]["local_path"] == path


def test_cli_auth_exposes_hub_token_lifecycle_only():
    runner = CliRunner()
    help_result = runner.invoke(app, ["auth", "--help"])
    removed_result = runner.invoke(app, ["auth", "token-create"])

    assert help_result.exit_code == 0
    for command in ("login", "logout", "switch", "list", "token", "whoami"):
        assert command in help_result.output
    assert removed_result.exit_code != 0
    assert "token-create" in removed_result.output


def test_cli_cache_help_is_registered():
    result = CliRunner().invoke(app, ["cache", "--help"])

    assert result.exit_code == 0
    assert "Manage the local MEGA cache directory." in result.output
    assert "verify" in result.output


def test_migrated_hub_download_url_and_filesystem_protocol(monkeypatch):
    monkeypatch.delenv("MEGA_ENDPOINT", raising=False)

    assert mega_hub_url("mega/demo", "weights/model.mega") == (
        "https://mega.tensorplay.cn/api/repos/mega/demo/resolve/weights/model.mega?revision=main"
    )
    assert MegaFileSystem.protocol == "mega"


def test_public_mega_api_reuses_native_service_client(monkeypatch):
    calls = []

    class RecordingClient:
        def create_repo(self, repo_id, **kwargs):
            calls.append(("create", repo_id, kwargs))
            return RepoInfo(
                repo_id=repo_id,
                private=kwargs["private"],
                created_at="now",
                updated_at="now",
                repo_type=kwargs["repo_type"],
            )

        def repo_info(self, repo_id):
            calls.append(("info", repo_id))
            return RepoInfo(
                repo_id=repo_id,
                private=False,
                created_at="now",
                updated_at="now",
                repo_type="dataset",
            )

        def update_repo(self, repo_id, **kwargs):
            calls.append(("update", repo_id, kwargs))

        def move_repo(self, from_id, to_id):
            calls.append(("move", from_id, to_id))

        def delete_repo(self, repo_id):
            calls.append(("delete", repo_id))

    client = RecordingClient()
    monkeypatch.setattr("megatensors._hub.mega_api._mega_service_client", lambda api, token=None: client)
    api = MegaApi(endpoint="https://hub.example.test", token="secret")

    url = api.create_repo("mega/demo", repo_type="dataset", private=True, exist_ok=True)
    api.update_repo_settings("mega/demo", repo_type="dataset", private=False)
    api.move_repo("mega/demo", "mega/moved", repo_type="dataset")
    api.delete_repo("mega/moved", repo_type="dataset")

    assert str(url) == "https://hub.example.test/datasets/mega/demo"
    assert calls == [
        ("create", "mega/demo", {"repo_type": "dataset", "private": True, "exist_ok": True}),
        ("info", "mega/demo"),
        ("update", "mega/demo", {"private": False}),
        ("info", "mega/demo"),
        ("move", "mega/demo", "mega/moved"),
        ("info", "mega/moved"),
        ("delete", "mega/moved"),
    ]


def test_public_mega_api_translates_native_not_found(monkeypatch):
    class MissingClient:
        def repo_info(self, repo_id):
            raise MegaHubError(
                "404 repository not found",
                status_code=404,
                method="GET",
                url=f"https://hub.example.test/api/repos/{repo_id}",
            )

    monkeypatch.setattr("megatensors._hub.mega_api._mega_service_client", lambda api, token=None: MissingClient())
    from megatensors._hub.errors import RepositoryNotFoundError

    with pytest.raises(RepositoryNotFoundError):
        MegaApi(endpoint="https://hub.example.test").repo_info("mega/missing")


def test_migrated_hub_uses_mega_token_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MEGA_TOKEN", raising=False)
    _configure_token_paths(tmp_path, monkeypatch)
    monkeypatch.setenv("MEGA_HOME", str(tmp_path))
    save_auth(token="mega-config-token", endpoint="https://hub.example.test")

    assert build_mega_headers()["authorization"] == "Bearer mega-config-token"


def test_filesystem_maps_missing_remote_repo_to_false(monkeypatch):
    from megatensors._hub.errors import RepositoryNotFoundError
    from megatensors._hub.mega_api import MegaApi

    def missing(self, repo_id, **kwargs):
        import httpx

        request = httpx.Request("GET", f"https://hub.example.test/api/repos/{repo_id}")
        response = httpx.Response(404, request=request)
        raise RepositoryNotFoundError("not found", response=response)

    monkeypatch.setattr(MegaApi, "repo_info", missing)
    fs = MegaFileSystem(endpoint="https://hub.example.test", skip_instance_cache=True)

    assert fs.exists("mega://mega/missing/file.bin") is False

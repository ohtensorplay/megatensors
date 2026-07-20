# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
import hashlib
import http.client
import json
import mimetypes
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from megatensors._jobs_client import JobsClientMixin
from megatensors._hub import constants as hub_constants
from megatensors._hub.utils import get_token
from megatensors._hub.utils._auth import _write_secret


DEFAULT_ENDPOINT = "https://mega.tensorplay.cn"
CONFIG_ENV_VAR = "MEGA_HOME"
ENDPOINT_ENV_VAR = "MEGA_ENDPOINT"
TOKEN_ENV_VAR = "MEGA_TOKEN"
MULTIPART_THRESHOLD = 16 * 1024 * 1024
_UNSET = object()


class MegaHubError(RuntimeError):
    """Raised when the MEGA Hub service rejects or cannot complete a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        method: str | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.method = method
        self.url = url


@dataclass(frozen=True)
class AuthConfig:
    endpoint: str
    token: Optional[str] = None


@dataclass(frozen=True)
class AccountKeyInfo:
    key_id: str
    key_type: str
    name: str
    public_key: str
    fingerprint: str
    created_at: str
    revoked_at: Optional[str] = None


@dataclass(frozen=True)
class WebhookDeliveryInfo:
    delivery_id: str
    webhook_id: str
    event: str
    state: str
    attempt_count: int
    response_status: Optional[int]
    error: Optional[str]
    created_at: str
    delivered_at: Optional[str]


@dataclass(frozen=True)
class WebhookLastDeliveryInfo:
    state: str
    response_status: Optional[int]
    error: Optional[str]
    delivered_at: str


@dataclass(frozen=True)
class WebhookInfo:
    webhook_id: str
    name: str
    url: str
    scope: str
    repo_id: Optional[str]
    events: tuple[str, ...]
    enabled: bool
    secret_configured: bool
    created_at: str
    updated_at: str
    last_delivery: Optional[WebhookLastDeliveryInfo] = None


@dataclass(frozen=True)
class CreatedWebhook:
    webhook: WebhookInfo
    signing_secret: str


def config_dir() -> Path:
    return _active_token_path().parent


def config_path() -> Path:
    return _active_token_path()


def load_auth() -> AuthConfig:
    endpoint = os.environ.get(ENDPOINT_ENV_VAR) or hub_constants.ENDPOINT or DEFAULT_ENDPOINT
    token = os.environ.get(TOKEN_ENV_VAR) or get_token()
    return AuthConfig(endpoint=_normalize_endpoint(endpoint), token=token)


def save_auth(*, token: str, endpoint: str) -> AuthConfig:
    cfg = AuthConfig(endpoint=_normalize_endpoint(endpoint), token=token)
    hub_constants.ENDPOINT = cfg.endpoint
    token_path = _active_token_path()
    hub_constants.MEGA_TOKEN_PATH = str(token_path)
    hub_constants.MEGA_STORED_TOKENS_PATH = str(token_path.parent / "stored_tokens")
    _write_secret(token_path, token)
    return cfg


def clear_auth() -> None:
    for path in {config_path(), Path(hub_constants.MEGA_TOKEN_PATH), Path(hub_constants.MEGA_STORED_TOKENS_PATH)}:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _active_token_path() -> Path:
    if token_path := os.environ.get("MEGA_TOKEN_PATH"):
        return Path(token_path).expanduser()
    if mega_home := os.environ.get(CONFIG_ENV_VAR):
        return Path(mega_home).expanduser() / "token"
    return Path(hub_constants.MEGA_TOKEN_PATH).expanduser()


@dataclass(frozen=True)
class RepoInfo:
    repo_id: str
    private: bool
    created_at: str
    updated_at: str
    owner: str = ""
    repo_type: str = "model"
    description: str = ""
    tags: tuple[str, ...] = ()
    license: str = ""
    likes: int = 0
    downloads: int = 0


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int
    sha256: str
    content_type: Optional[str] = None


@dataclass(frozen=True)
class CommitInfo:
    revision: str
    parent_revision: Optional[str]
    message: str
    author: str
    created_at: str
    signature_status: Optional[str] = None
    signer_fingerprint: Optional[str] = None
    signer_subject: Optional[str] = None
    author_email: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class RefInfo:
    name: str
    ref: str
    target_revision: str
    message: Optional[str] = None
    updated_at: str = ""


@dataclass(frozen=True)
class RepoRefs:
    branches: tuple[RefInfo, ...]
    tags: tuple[RefInfo, ...]


@dataclass(frozen=True)
class CollectionOwnerInfo:
    handle: str
    display_name: str
    kind: str


@dataclass(frozen=True)
class CollectionItemInfo:
    item_object_id: str
    item_id: str
    item_type: str
    title: str
    description: str
    href: str
    note: str
    position: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CollectionInfo:
    id: str
    slug: str
    title: str
    description: str
    private: bool
    theme: str
    position: int
    owner: CollectionOwnerInfo
    item_count: int
    upvotes: int
    upvoted: bool
    items: tuple[CollectionItemInfo, ...]
    created_at: str
    updated_at: str
    url: str


@dataclass(frozen=True)
class CommitChange:
    path: str
    change: str
    size: int
    sha256: str
    previous_sha256: Optional[str] = None


@dataclass(frozen=True)
class CommitDetail:
    revision: str
    parent_revision: Optional[str]
    message: str
    author: str
    created_at: str
    files: tuple[CommitChange, ...]
    signature_status: Optional[str] = None
    signer_fingerprint: Optional[str] = None
    signer_subject: Optional[str] = None
    author_email: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class CommunityAuthor:
    handle: str
    display_name: str
    avatar_url: Optional[str] = None


@dataclass(frozen=True)
class PullRequestInfo:
    source_branch: str
    target_branch: str
    source_revision: str
    target_revision: str
    merged_at: Optional[str] = None
    merged_by: Optional[str] = None


@dataclass(frozen=True)
class DiscussionInfo:
    repo_id: str
    repo_type: str
    number: int
    title: str
    kind: str
    status: str
    author: CommunityAuthor
    reply_count: int
    reaction_count: int
    created_at: str
    updated_at: str
    endpoint: str
    pull_request: Optional[PullRequestInfo] = None

    @property
    def num(self) -> int:
        """HF-compatible alias used by existing community integrations."""
        return self.number

    @property
    def is_pull_request(self) -> bool:
        return self.kind == "pull_request"

    @property
    def url(self) -> str:
        prefix = "" if self.repo_type == "model" else f"/{self.repo_type}s"
        return f"{self.endpoint}{prefix}/{self.repo_id}/discussions/{self.number}"


@dataclass(frozen=True)
class DiscussionMessageInfo:
    message_id: str
    author: CommunityAuthor
    body: str
    is_original: bool
    created_at: str
    updated_at: str
    reaction_count: int
    viewer_reacted: bool
    can_edit: bool
    can_delete: bool


@dataclass(frozen=True)
class DiscussionPermissions:
    can_reply: bool
    can_close: bool
    can_reopen: bool
    can_delete: bool
    can_merge: bool
    merge_blocked_reason: Optional[str] = None


@dataclass(frozen=True)
class DiscussionThread:
    discussion: DiscussionInfo
    messages: tuple[DiscussionMessageInfo, ...]
    permissions: DiscussionPermissions


@dataclass(frozen=True)
class DiscussionPage:
    discussions: tuple[DiscussionInfo, ...]
    counts: Mapping[str, int]
    status: str
    kind: str
    sort: str
    query: str
    page: int
    limit: int
    has_more: bool


class MegaHubClient(JobsClientMixin):
    """Route-level client for the native MEGA Hub service API.

    This client owns transport and Worker route details. Higher-level Python
    workflows live in :class:`megatensors.MegaApi` and reuse this client.
    """

    def __init__(self, endpoint: Optional[str] = None, token: Optional[str] | bool = None) -> None:
        auth = load_auth()
        self.endpoint = _normalize_endpoint(endpoint or auth.endpoint)
        self.token = None if token is False else token if isinstance(token, str) else auth.token

    def whoami(self) -> Mapping[str, Any]:
        return self._request_json("GET", "/api/whoami", auth=True)

    def list_account_keys(self) -> list[AccountKeyInfo]:
        """List active SSH and GPG public keys on the current account."""
        data = self._request_json("GET", "/api/me/keys", auth=True)
        return [_account_key_info(item) for item in data.get("keys", [])]

    def add_account_key(self, *, key_type: str, name: str, public_key: str) -> AccountKeyInfo:
        """Register a public key without ever accepting private-key material."""
        normalized_type = key_type.strip().lower()
        if normalized_type not in {"ssh", "gpg"}:
            raise ValueError("key_type must be 'ssh' or 'gpg'")
        _reject_private_key(public_key)
        data = self._request_json(
            "POST",
            "/api/me/keys",
            json_body={"key_type": normalized_type, "name": name, "public_key": public_key},
            auth=True,
        )
        return _account_key_info(data)

    def delete_account_key(self, key_id: str) -> None:
        """Remove an active public key from the current account."""
        self._request("DELETE", f"/api/me/keys/{urllib.parse.quote(key_id, safe='')}", auth=True)

    def list_webhooks(self) -> list[WebhookInfo]:
        """List webhook routes owned by the current MEGA account."""
        data = self._request_json("GET", "/api/me/webhooks", auth=True)
        return [_webhook_info(item) for item in data.get("webhooks", [])]

    def get_webhook(self, webhook_id: str) -> WebhookInfo:
        """Return one webhook route owned by the current MEGA account."""
        data = self._request_json(
            "GET",
            f"/api/me/webhooks/{urllib.parse.quote(webhook_id, safe='')}",
            auth=True,
        )
        return _webhook_info(data["webhook"])

    def create_webhook(
        self,
        *,
        name: str,
        url: str,
        events: Iterable[str],
        scope: str = "account",
        repo_id: Optional[str] = None,
        secret: Optional[str] = None,
    ) -> CreatedWebhook:
        """Create a signed MEGA webhook route and return its one-time signing secret."""
        body: dict[str, Any] = {
            "name": name,
            "url": url,
            "events": list(events),
            "scope": scope,
        }
        if repo_id is not None:
            body["repo_id"] = repo_id
        if secret is not None:
            body["secret"] = secret
        data = self._request_json("POST", "/api/me/webhooks", json_body=body, auth=True)
        return CreatedWebhook(webhook=_webhook_info(data["webhook"]), signing_secret=str(data["signing_secret"]))

    def update_webhook(
        self,
        webhook_id: str,
        *,
        name: Optional[str] = None,
        url: Optional[str] = None,
        events: Optional[Iterable[str]] = None,
        enabled: Optional[bool] = None,
        secret: Optional[str] = None,
    ) -> WebhookInfo:
        """Update only the supplied fields of one webhook route."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if url is not None:
            body["url"] = url
        if events is not None:
            body["events"] = list(events)
        if enabled is not None:
            body["enabled"] = enabled
        if secret is not None:
            body["secret"] = secret
        if not body:
            raise ValueError("provide at least one webhook field to update")
        data = self._request_json(
            "PATCH",
            f"/api/me/webhooks/{urllib.parse.quote(webhook_id, safe='')}",
            json_body=body,
            auth=True,
        )
        return _webhook_info(data["webhook"])

    def test_webhook(self, webhook_id: str) -> WebhookDeliveryInfo:
        """Send and record a signed test payload for a webhook route."""
        data = self._request_json(
            "POST",
            f"/api/me/webhooks/{urllib.parse.quote(webhook_id, safe='')}/test",
            auth=True,
        )
        return _webhook_delivery_info(data["delivery"])

    def list_webhook_deliveries(self, webhook_id: str, *, limit: int = 20) -> list[WebhookDeliveryInfo]:
        """Return retained delivery receipts for one webhook route."""
        safe_limit = min(max(int(limit), 1), 100)
        data = self._request_json(
            "GET",
            f"/api/me/webhooks/{urllib.parse.quote(webhook_id, safe='')}/deliveries?limit={safe_limit}",
            auth=True,
        )
        return [_webhook_delivery_info(item) for item in data.get("deliveries", [])]

    def delete_webhook(self, webhook_id: str) -> None:
        """Delete a webhook route and its retained delivery history."""
        self._request("DELETE", f"/api/me/webhooks/{urllib.parse.quote(webhook_id, safe='')}", auth=True)

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        private: bool = False,
        exist_ok: bool = False,
        description: str = "",
        tags: Optional[Iterable[str]] = None,
        license: str = "",
    ) -> RepoInfo:
        data = self._request_json(
            "POST",
            "/api/repos",
            json_body={
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": exist_ok,
                "description": description,
                "tags": list(tags or []),
                "license": license,
            },
            auth=True,
        )
        return _repo_info(data)

    def list_repos(
        self,
        *,
        limit: int = 100,
        repo_type: Optional[str] = None,
        owner: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list[RepoInfo]:
        remaining = int(limit)
        if remaining <= 0:
            return []
        repos: list[RepoInfo] = []
        cursor: Optional[str] = None
        while remaining > 0:
            query = {"limit": str(min(remaining, 500))}
            if repo_type:
                query["type"] = repo_type
            if owner:
                query["owner"] = owner
            if search:
                query["q"] = search
            if cursor:
                query["cursor"] = cursor
            data = self._request_json("GET", f"/api/repos?{urllib.parse.urlencode(query)}")
            page = [_repo_info(item) for item in data.get("repos", [])]
            repos.extend(page)
            remaining -= len(page)
            next_cursor = data.get("next_cursor")
            if not page or not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
                return repos
            cursor = next_cursor
        return repos

    def repo_info(self, repo_id: str) -> RepoInfo:
        data = self._request_json("GET", f"/api/repos/{_quote_repo_id(repo_id)}")
        return _repo_info(data)

    def delete_repo(self, repo_id: str) -> None:
        self._request("DELETE", f"/api/repos/{_quote_repo_id(repo_id)}", auth=True)

    def move_repo(self, from_id: str, to_id: str) -> RepoInfo:
        data = self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(from_id)}/move",
            json_body={"repo_id": to_id},
            auth=True,
        )
        return _repo_info(data)

    def duplicate_repo(self, from_id: str, to_id: str, *, private: Optional[bool] = None) -> RepoInfo:
        body: dict[str, Any] = {"repo_id": to_id}
        if private is not None:
            body["private"] = private
        data = self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(from_id)}/duplicate",
            json_body=body,
            auth=True,
        )
        return _repo_info(data)

    def update_repo(
        self,
        repo_id: str,
        *,
        private: Optional[bool] = None,
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        license: Optional[str] = None,
    ) -> RepoInfo:
        body: dict[str, Any] = {}
        if private is not None:
            body["private"] = private
        if description is not None:
            body["description"] = description
        if tags is not None:
            body["tags"] = list(tags)
        if license is not None:
            body["license"] = license
        if not body:
            raise ValueError("provide at least one repository field to update")
        data = self._request_json(
            "PATCH",
            f"/api/repos/{_quote_repo_id(repo_id)}",
            json_body=body,
            auth=True,
        )
        return _repo_info(data)

    def list_collections(
        self,
        *,
        owner: Optional[str] = None,
        item: Optional[str] = None,
        sort: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 20,
    ) -> list[CollectionInfo]:
        """List collections visible to the current identity."""
        remaining = int(limit)
        if remaining <= 0:
            return []
        collections: list[CollectionInfo] = []
        cursor: Optional[str] = None
        while remaining > 0:
            query = {"limit": str(min(remaining, 100))}
            if owner:
                query["owner"] = owner
            if item:
                query["item"] = item
            if sort:
                query["sort"] = sort
            if search:
                query["q"] = search
            if cursor:
                query["cursor"] = cursor
            data = self._request_json(
                "GET", f"/api/collections?{urllib.parse.urlencode(query)}"
            )
            page = [
                _collection_info(value, endpoint=self.endpoint)
                for value in data.get("collections", [])
                if isinstance(value, Mapping)
            ]
            collections.extend(page)
            remaining -= len(page)
            next_cursor = data.get("next_cursor")
            if not page or not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
                return collections
            cursor = next_cursor
        return collections

    def get_collection(self, collection_slug: str) -> CollectionInfo:
        data = self._request_json(
            "GET", f"/api/collections/{_quote_collection_slug(collection_slug)}"
        )
        return _collection_info(data, endpoint=self.endpoint)

    def create_collection(
        self,
        title: str,
        *,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        private: bool = False,
        theme: Optional[str] = None,
        exists_ok: bool = False,
    ) -> CollectionInfo:
        body: dict[str, Any] = {
            "title": title,
            "private": private,
            "exists_ok": exists_ok,
        }
        if namespace is not None:
            body["namespace"] = namespace
        if description is not None:
            body["description"] = description
        if theme is not None:
            body["theme"] = theme
        data = self._request_json(
            "POST", "/api/collections", json_body=body, auth=True
        )
        return _collection_info(data, endpoint=self.endpoint)

    def update_collection(
        self,
        collection_slug: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        position: Optional[int] = None,
        private: Optional[bool] = None,
        theme: Optional[str] = None,
    ) -> CollectionInfo:
        body = {
            key: value
            for key, value in {
                "title": title,
                "description": description,
                "position": position,
                "private": private,
                "theme": theme,
            }.items()
            if value is not None
        }
        if not body:
            raise ValueError("provide at least one collection field to update")
        data = self._request_json(
            "PATCH",
            f"/api/collections/{_quote_collection_slug(collection_slug)}",
            json_body=body,
            auth=True,
        )
        return _collection_info(data, endpoint=self.endpoint)

    def delete_collection(self, collection_slug: str, *, missing_ok: bool = False) -> None:
        try:
            self._request(
                "DELETE",
                f"/api/collections/{_quote_collection_slug(collection_slug)}",
                auth=True,
            )
        except MegaHubError as error:
            if missing_ok and error.status_code == 404:
                return
            raise

    def add_collection_item(
        self,
        collection_slug: str,
        item_id: str,
        item_type: str,
        *,
        note: Optional[str] = None,
        position: Optional[int] = None,
        exists_ok: bool = False,
    ) -> CollectionInfo:
        body: dict[str, Any] = {
            "item": {"id": item_id, "type": item_type},
            "exists_ok": exists_ok,
        }
        if note is not None:
            body["note"] = note
        if position is not None:
            body["position"] = position
        data = self._request_json(
            "POST",
            f"/api/collections/{_quote_collection_slug(collection_slug)}/items",
            json_body=body,
            auth=True,
        )
        return _collection_info(data, endpoint=self.endpoint)

    def update_collection_item(
        self,
        collection_slug: str,
        item_object_id: str,
        *,
        note: Optional[str] = None,
        position: Optional[int] = None,
    ) -> CollectionInfo:
        body = {
            key: value
            for key, value in {"note": note, "position": position}.items()
            if value is not None
        }
        if not body:
            raise ValueError("provide a note or position to update")
        data = self._request_json(
            "PATCH",
            f"/api/collections/{_quote_collection_slug(collection_slug)}/items/{urllib.parse.quote(item_object_id, safe='')}",
            json_body=body,
            auth=True,
        )
        return _collection_info(data, endpoint=self.endpoint)

    def delete_collection_item(
        self,
        collection_slug: str,
        item_object_id: str,
        *,
        missing_ok: bool = False,
    ) -> CollectionInfo:
        suffix = "?missing_ok=true" if missing_ok else ""
        data = self._request_json(
            "DELETE",
            f"/api/collections/{_quote_collection_slug(collection_slug)}/items/{urllib.parse.quote(item_object_id, safe='')}{suffix}",
            auth=True,
        )
        return _collection_info(data, endpoint=self.endpoint)

    def list_discussions(
        self,
        repo_id: str,
        *,
        status: str = "open",
        kind: str = "all",
        sort: str = "recently-created",
        search: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
        repo_type: str = "model",
    ) -> DiscussionPage:
        """List one page of repository discussions and pull requests."""
        _validate_repo_type(repo_type)
        if status not in {"all", "open", "closed"}:
            raise ValueError("status must be 'all', 'open', or 'closed'")
        if kind not in {"all", "discussion", "pull_request"}:
            raise ValueError("kind must be 'all', 'discussion', or 'pull_request'")
        if sort not in {"recently-created", "recently-updated"}:
            raise ValueError("sort must be 'recently-created' or 'recently-updated'")
        if page < 1:
            raise ValueError("page must be at least 1")
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        query = {
            "status": status,
            "kind": kind,
            "sort": sort,
            "page": str(page),
            "limit": str(limit),
        }
        if search:
            query["q"] = search
        data = self._request_json(
            "GET",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions?{urllib.parse.urlencode(query)}",
        )
        return _discussion_page(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def get_discussion(
        self,
        repo_id: str,
        number: int,
        *,
        repo_type: str = "model",
    ) -> DiscussionThread:
        """Return a complete discussion thread with viewer permissions."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        data = self._request_json(
            "GET",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}",
        )
        return _discussion_thread(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def create_discussion(
        self,
        repo_id: str,
        *,
        title: str,
        body: str,
        kind: str = "discussion",
        source_branch: Optional[str] = None,
        target_branch: str = "main",
        repo_type: str = "model",
    ) -> DiscussionThread:
        """Open a discussion or a branch-backed pull request."""
        _validate_repo_type(repo_type)
        if kind not in {"discussion", "pull_request"}:
            raise ValueError("kind must be 'discussion' or 'pull_request'")
        payload: dict[str, Any] = {"title": title, "body": body, "kind": kind}
        if kind == "pull_request":
            if not source_branch:
                raise ValueError("source_branch is required for a pull request")
            payload.update(source_branch=source_branch, target_branch=target_branch)
        elif source_branch is not None:
            raise ValueError("source_branch is only valid for a pull request")
        data = self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions",
            json_body=payload,
            auth=True,
        )
        return _discussion_thread(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def reply_to_discussion(
        self,
        repo_id: str,
        number: int,
        body: str,
        *,
        repo_type: str = "model",
    ) -> DiscussionThread:
        """Append one Markdown message to an open discussion."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        data = self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}/messages",
            json_body={"body": body},
            auth=True,
        )
        return _discussion_thread(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def update_discussion(
        self,
        repo_id: str,
        number: int,
        *,
        status: Optional[str] = None,
        title: Optional[str] = None,
        comment: Optional[str] = None,
        repo_type: str = "model",
    ) -> DiscussionThread:
        """Rename, close, or reopen a discussion through the native Worker contract."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        payload: dict[str, Any] = {}
        if status is not None:
            if status not in {"open", "closed"}:
                raise ValueError("status must be 'open' or 'closed'")
            payload["status"] = status
        if title is not None:
            payload["title"] = title
        if comment is not None:
            payload["comment"] = comment
        if not payload or (set(payload) == {"comment"}):
            raise ValueError("provide a status or title to update")
        data = self._request_json(
            "PATCH",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}",
            json_body=payload,
            auth=True,
        )
        return _discussion_thread(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def merge_pull_request(
        self,
        repo_id: str,
        number: int,
        *,
        comment: Optional[str] = None,
        repo_type: str = "model",
    ) -> DiscussionThread:
        """Fast-forward a ready MEGA pull request."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        data = self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}/merge",
            json_body={"comment": comment} if comment is not None else {},
            auth=True,
        )
        return _discussion_thread(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def delete_discussion(
        self,
        repo_id: str,
        number: int,
        *,
        repo_type: str = "model",
    ) -> None:
        """Delete a non-merged discussion owned or moderated by the caller."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        self._request(
            "DELETE",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}",
            auth=True,
        )

    def edit_discussion_message(
        self,
        repo_id: str,
        number: int,
        message_id: str,
        body: str,
        *,
        repo_type: str = "model",
    ) -> DiscussionThread:
        """Edit a discussion message owned or moderated by the caller."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        data = self._request_json(
            "PATCH",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}/messages/{urllib.parse.quote(message_id, safe='')}",
            json_body={"body": body},
            auth=True,
        )
        return _discussion_thread(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def delete_discussion_message(
        self,
        repo_id: str,
        number: int,
        message_id: str,
        *,
        repo_type: str = "model",
    ) -> DiscussionThread:
        """Delete a non-original discussion message."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        data = self._request_json(
            "DELETE",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}/messages/{urllib.parse.quote(message_id, safe='')}",
            auth=True,
        )
        return _discussion_thread(data, repo_id=repo_id, repo_type=repo_type, endpoint=self.endpoint)

    def set_discussion_reaction(
        self,
        repo_id: str,
        number: int,
        message_id: str,
        *,
        active: bool = True,
        repo_type: str = "model",
    ) -> Mapping[str, Any]:
        """Add or remove the MEGA ``fire`` reaction on a discussion message."""
        _validate_discussion_number(number)
        _validate_repo_type(repo_type)
        data = self._request_json(
            "PUT" if active else "DELETE",
            f"/api/repos/{_quote_repo_id(repo_id)}/discussions/{number}/messages/{urllib.parse.quote(message_id, safe='')}/reactions/fire",
            auth=True,
        )
        reaction = data.get("reaction")
        if not isinstance(reaction, Mapping):
            raise MegaHubError("expected discussion reaction response")
        return reaction

    def list_files(self, repo_id: str, *, revision: str = "main") -> list[FileInfo]:
        files: list[FileInfo] = []
        cursor: Optional[str] = None
        resolved_revision = revision
        while True:
            query = {"revision": resolved_revision, "limit": "1000"}
            if cursor:
                query["cursor"] = cursor
            data = self._request_json(
                "GET",
                f"/api/repos/{_quote_repo_id(repo_id)}/files?{urllib.parse.urlencode(query)}",
            )
            files.extend(_file_info(item) for item in data.get("files", []))
            resolved_revision = str(data.get("revision", resolved_revision))
            next_cursor = data.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
                return files
            cursor = next_cursor

    def delete_file(self, repo_id: str, path_in_repo: str, *, revision: str = "main", commit_message: Optional[str] = None) -> Mapping[str, Any]:
        query = urllib.parse.urlencode({"revision": revision, "commit_message": commit_message or f"Delete {path_in_repo}"})
        return self._request_json(
            "DELETE",
            f"/api/repos/{_quote_repo_id(repo_id)}/files/{_quote_path(path_in_repo)}?{query}",
            auth=True,
        )

    def list_refs(self, repo_id: str) -> RepoRefs:
        data = self._request_json("GET", f"/api/repos/{_quote_repo_id(repo_id)}/refs")
        return RepoRefs(
            branches=tuple(_ref_info(item) for item in data.get("branches", [])),
            tags=tuple(_ref_info(item) for item in data.get("tags", [])),
        )

    def create_branch(
        self,
        repo_id: str,
        branch: str,
        *,
        revision: str = "main",
        exist_ok: bool = False,
    ) -> RefInfo:
        data = self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(repo_id)}/refs/branches",
            json_body={"name": branch, "revision": revision, "exist_ok": exist_ok},
            auth=True,
        )
        return _ref_info(data)

    def delete_branch(self, repo_id: str, branch: str) -> None:
        self._request(
            "DELETE",
            f"/api/repos/{_quote_repo_id(repo_id)}/refs/branches/{urllib.parse.quote(branch, safe='')}",
            auth=True,
        )

    def create_tag(
        self,
        repo_id: str,
        tag: str,
        *,
        revision: str = "main",
        message: Optional[str] = None,
        exist_ok: bool = False,
    ) -> RefInfo:
        data = self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(repo_id)}/refs/tags",
            json_body={"name": tag, "revision": revision, "message": message or "", "exist_ok": exist_ok},
            auth=True,
        )
        return _ref_info(data)

    def delete_tag(self, repo_id: str, tag: str) -> None:
        self._request(
            "DELETE",
            f"/api/repos/{_quote_repo_id(repo_id)}/refs/tags/{urllib.parse.quote(tag, safe='')}",
            auth=True,
        )

    def list_commits(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> tuple[list[CommitInfo], Optional[str]]:
        query = {"limit": str(limit)}
        query["revision"] = revision
        if cursor:
            query["cursor"] = cursor
        data = self._request_json("GET", f"/api/repos/{_quote_repo_id(repo_id)}/commits?{urllib.parse.urlencode(query)}")
        commits = [
            CommitInfo(
                revision=str(item["revision"]),
                parent_revision=item.get("parent_revision"),
                message=str(item.get("message", "")),
                author=str(item.get("author", "")),
                created_at=str(item.get("created_at", "")),
                signature_status=item.get("signature_status"),
                signer_fingerprint=item.get("signer_fingerprint"),
                signer_subject=item.get("signer_subject"),
                author_email=item.get("author_email"),
                description=item.get("description"),
            )
            for item in data.get("commits", [])
        ]
        return commits, data.get("next_cursor")

    def get_commit(self, repo_id: str, revision: str) -> CommitDetail:
        data = self._request_json(
            "GET",
            f"/api/repos/{_quote_repo_id(repo_id)}/commits/{urllib.parse.quote(revision, safe='')}",
        )
        return CommitDetail(
            revision=str(data["revision"]),
            parent_revision=data.get("parent_revision"),
            message=str(data.get("message", "")),
            author=str(data.get("author", "")),
            created_at=str(data.get("created_at", "")),
            files=tuple(
                CommitChange(
                    path=str(item["path"]),
                    change=str(item["change"]),
                    size=int(item.get("size", 0)),
                    sha256=str(item.get("sha256", "")),
                    previous_sha256=item.get("previous_sha256"),
                )
                for item in data.get("files", [])
            ),
            signature_status=data.get("signature_status"),
            signer_fingerprint=data.get("signer_fingerprint"),
            signer_subject=data.get("signer_subject"),
            author_email=data.get("author_email"),
            description=data.get("description"),
        )

    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[Mapping[str, Any]],
        *,
        revision: str = "main",
        parent_revision: str | None | object = _UNSET,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Apply file operations as one repository commit.

        Passing ``parent_revision`` enables compare-and-swap protection. ``None``
        explicitly means that the destination branch must not exist yet.
        """
        body: dict[str, Any] = {
            "revision": revision,
            "commit_message": commit_message or "Update repository",
            "operations": [dict(operation) for operation in operations],
        }
        if parent_revision is not _UNSET:
            body["parent_revision"] = parent_revision
        if commit_description is not None:
            body["commit_description"] = commit_description
        return self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id(repo_id)}/commits",
            json_body=body,
            auth=True,
        )

    def create_token(
        self, *, subject: str, role: str = "write", name: str = "CLI token", expires_at: Optional[str] = None
    ) -> Mapping[str, Any]:
        body: dict[str, Any] = {"subject": subject, "role": role, "name": name}
        if expires_at is not None:
            body["expires_at"] = expires_at
        return self._request_json("POST", "/api/tokens", json_body=body, auth=True)

    def list_tokens(self) -> list[Mapping[str, Any]]:
        return list(self._request_json("GET", "/api/tokens", auth=True).get("tokens", []))

    def revoke_token(self, token_id: str) -> None:
        self._request("DELETE", f"/api/tokens/{urllib.parse.quote(token_id, safe='')}", auth=True)

    def list_audit_events(self, *, limit: int = 50, cursor: Optional[str] = None) -> tuple[list[Mapping[str, Any]], Optional[str]]:
        query = {"limit": str(limit)}
        if cursor:
            query["cursor"] = cursor
        data = self._request_json("GET", f"/api/audit?{urllib.parse.urlencode(query)}", auth=True)
        return list(data.get("events", [])), data.get("next_cursor")

    def upload_file(
        self,
        repo_id: str,
        local_path: str | Path,
        *,
        path_in_repo: Optional[str] = None,
        revision: str = "main",
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        private: bool = False,
        repo_type: str = "model",
    ) -> Mapping[str, Any]:
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        remote_path = _clean_repo_path(path_in_repo or path.name)
        query_params = {
            "revision": revision,
            "commit_message": commit_message or f"Upload {remote_path}",
        }
        query = urllib.parse.urlencode(query_params)
        sha256 = _sha256_file(path)
        if commit_description is not None:
            operation = self._stage_file(
                repo_id,
                path,
                remote_path=remote_path,
                revision=revision,
                sha256=sha256,
            )
            committed = self.create_commit(
                repo_id,
                [operation],
                revision=revision,
                commit_message=commit_message or f"Upload {remote_path}",
                commit_description=commit_description,
            )
            return {
                "repo_id": repo_id,
                "path": remote_path,
                "revision": str(committed["revision"]),
                "size": int(operation["size"]),
                "sha256": str(operation["sha256"]),
            }
        if path.stat().st_size >= MULTIPART_THRESHOLD:
            return self._upload_file_resumable(
                repo_id,
                path,
                remote_path=remote_path,
                revision=revision,
                commit_message=commit_message or f"Upload {remote_path}",
                sha256=sha256,
            )
        headers = {
            "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "Content-Length": str(path.stat().st_size),
            "X-Mega-Sha256": sha256,
            "X-Mega-Size": str(path.stat().st_size),
        }
        with path.open("rb") as f:
            return self._request_json(
                "PUT",
                f"/api/repos/{_quote_repo_id(repo_id)}/files/{_quote_path(remote_path)}?{query}",
                data=f,
                headers=headers,
                auth=True,
            )

    def copy_file(
        self,
        source_repo_id: str,
        source_path: str,
        destination_repo_id: str,
        destination_path: str,
        *,
        source_revision: str = "main",
        revision: str = "main",
        commit_message: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Copy one repository file server-side without downloading its bytes."""
        return self.copy_files(
            source_repo_id,
            source_path,
            destination_repo_id,
            destination_path,
            source_revision=source_revision,
            revision=revision,
            commit_message=commit_message,
        )

    def copy_files(
        self,
        source_repo_id: str,
        source_path: str,
        destination_repo_id: str,
        destination_path: str,
        *,
        source_revision: str = "main",
        revision: str = "main",
        source_merge_contents: bool = False,
        destination_is_directory: bool = False,
        commit_message: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Copy a repository file or directory server-side without transferring object bytes."""
        source_repo = _split_repo_id(source_repo_id)
        destination_repo = _split_repo_id(destination_repo_id)
        clean_source_path = _clean_repo_copy_path(source_path)
        clean_destination_path = _clean_repo_copy_path(destination_path)
        return self._request_json(
            "POST",
            f"/api/repos/{_quote_repo_id('/'.join(destination_repo))}/copy",
            json_body={
                "source_repo_id": "/".join(source_repo),
                "source_path": clean_source_path,
                "source_revision": source_revision,
                "path": clean_destination_path,
                "revision": revision,
                "source_merge_contents": source_merge_contents,
                "destination_is_directory": destination_is_directory,
                "commit_message": commit_message or f"Copy {source_repo_id}/{clean_source_path}",
            },
            auth=True,
        )

    def _upload_file_resumable(
        self,
        repo_id: str,
        path: Path,
        *,
        remote_path: str,
        revision: str,
        commit_message: str,
        sha256: str,
        stage_only: bool = False,
    ) -> Mapping[str, Any]:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        state_path = _upload_state_path(repo_id, remote_path, revision, size, sha256, stage_only=stage_only)
        state = _load_upload_state(state_path)
        if not _matches_upload_state(
            state,
            self.endpoint,
            repo_id,
            remote_path,
            revision,
            size,
            sha256,
            stage_only=stage_only,
        ):
            started = self._request_json(
                "POST",
                "/api/uploads",
                json_body={
                    "repo_id": repo_id,
                    "path": remote_path,
                    "revision": revision,
                    "commit_message": commit_message,
                    "size": size,
                    "sha256": sha256,
                    "content_type": content_type,
                    "stage_only": stage_only,
                },
                auth=True,
            )
            state = {
                "endpoint": self.endpoint,
                "repo_id": repo_id,
                "path": remote_path,
                "revision": revision,
                "size": size,
                "sha256": sha256,
                "stage_only": stage_only,
                "upload_id": str(started["upload_id"]),
                "part_size": int(started["part_size"]),
            }
            _save_upload_state(state_path, state)

        upload_id = str(state["upload_id"])
        part_size = int(state["part_size"])
        status = self._request_json("GET", f"/api/uploads/{urllib.parse.quote(upload_id, safe='')}", auth=True)
        completed = {int(part["part_number"]): str(part["etag"]) for part in status.get("parts", [])}
        total_parts = (size + part_size - 1) // part_size
        try:
            with path.open("rb") as file:
                for part_number in range(1, total_parts + 1):
                    offset = (part_number - 1) * part_size
                    length = min(part_size, size - offset)
                    if part_number in completed:
                        continue
                    file.seek(offset)
                    response = self._request_json(
                        "PUT",
                        f"/api/uploads/{urllib.parse.quote(upload_id, safe='')}/parts/{part_number}",
                        data=_FileSlice(file, length),
                        headers={"Content-Type": "application/octet-stream", "Content-Length": str(length)},
                        auth=True,
                    )
                    completed[part_number] = str(response["etag"])
            result = self._request_json(
                "POST",
                f"/api/uploads/{urllib.parse.quote(upload_id, safe='')}/complete",
                json_body={
                    "parts": [{"part_number": number, "etag": completed[number]} for number in range(1, total_parts + 1)]
                },
                auth=True,
            )
        except Exception:
            # Keep the session metadata so a later invocation resumes confirmed parts.
            raise
        else:
            state_path.unlink(missing_ok=True)
            return result

    def _stage_file(
        self,
        repo_id: str,
        path: Path,
        *,
        remote_path: str,
        revision: str,
        sha256: Optional[str] = None,
    ) -> Mapping[str, Any]:
        digest = sha256 or _sha256_file(path)
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if size >= MULTIPART_THRESHOLD:
            self._upload_file_resumable(
                repo_id,
                path,
                remote_path=remote_path,
                revision=revision,
                commit_message=f"Stage {remote_path}",
                sha256=digest,
                stage_only=True,
            )
        else:
            headers = {
                "Content-Type": content_type,
                "Content-Length": str(size),
                "X-Mega-Size": str(size),
            }
            with path.open("rb") as file:
                self._request_json(
                    "PUT",
                    f"/api/repos/{_quote_repo_id(repo_id)}/blobs/{digest}",
                    data=file,
                    headers=headers,
                    auth=True,
                )
        return {
            "operation": "add",
            "path": remote_path,
            "size": size,
            "sha256": digest,
            "content_type": content_type,
        }

    def upload_folder(
        self,
        repo_id: str,
        folder_path: str | Path,
        *,
        path_in_repo: str = "",
        revision: str = "main",
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        private: bool = False,
        repo_type: str = "model",
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        max_workers: Optional[int] = None,
        delete_missing: bool = False,
    ) -> list[Mapping[str, Any]]:
        root = Path(folder_path)
        if not root.is_dir():
            raise NotADirectoryError(root)
        files = list(_iter_upload_files(root, include=include, exclude=exclude))
        self.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        if not files and not delete_missing:
            return []
        refs = self.list_refs(repo_id)
        expected_parent = next(
            (ref.target_revision for ref in refs.branches if ref.name == revision),
            None,
        )
        if expected_parent is None:
            raise MegaHubError(f"branch not found: {revision}")

        remote_paths: set[str] = set()
        if delete_missing:
            prefix = path_in_repo.strip("/")
            remote_paths = {
                item.path
                for item in self.list_files(repo_id, revision=expected_parent)
                if not prefix or item.path == prefix or item.path.startswith(f"{prefix}/")
            }

        def stage_one(file_path: Path) -> Mapping[str, Any]:
            rel = file_path.relative_to(root).as_posix()
            remote_path = _join_repo_path(path_in_repo, rel)
            return self._stage_file(
                repo_id,
                file_path,
                remote_path=remote_path,
                revision=revision,
            )

        workers = max_workers or min(8, max(1, len(files)))
        if not files:
            operations = []
        elif workers <= 1:
            operations = [stage_one(file_path) for file_path in files]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                operations = list(executor.map(stage_one, files))
        if delete_missing:
            uploaded_paths = {str(operation["path"]) for operation in operations}
            operations.extend(
                {"operation": "delete", "path": path}
                for path in sorted(remote_paths - uploaded_paths)
            )
        if not operations:
            return []
        commit_options: dict[str, Any] = {
            "revision": revision,
            "parent_revision": expected_parent,
            "commit_message": commit_message or (
                f"Sync {len(files)} files" if delete_missing else f"Upload {len(files)} files"
            ),
        }
        if commit_description is not None:
            commit_options["commit_description"] = commit_description
        committed = self.create_commit(repo_id, operations, **commit_options)
        commit_revision = str(committed["revision"])
        return [
            {
                "repo_id": repo_id,
                "path": str(operation["path"]),
                "revision": commit_revision,
                "size": int(operation["size"]),
                "sha256": str(operation["sha256"]),
            }
            for operation in operations
            if operation["operation"] == "add"
        ]

    def iter_snapshot_files(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> list[FileInfo]:
        return [
            info
            for info in self.list_files(repo_id, revision=revision)
            if _matches_patterns(info.path, include=include, exclude=exclude)
        ]

    def download_files(
        self,
        repo_id: str,
        filenames: Iterable[str],
        *,
        local_dir: str | Path = ".",
        revision: str = "main",
        force: bool = False,
        max_workers: int = 1,
    ) -> list[Path]:
        names = list(filenames)
        workers = max_workers or 1
        if workers <= 1:
            return [
                self.download_file(repo_id, filename, local_dir=local_dir, revision=revision, force=force)
                for filename in names
            ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda filename: self.download_file(
                        repo_id,
                        filename,
                        local_dir=local_dir,
                        revision=revision,
                        force=force,
                    ),
                    names,
                )
            )

    def download_file(
        self,
        repo_id: str,
        filename: str,
        *,
        local_dir: str | Path = ".",
        revision: str = "main",
        force: bool = False,
    ) -> Path:
        remote_path = _clean_repo_path(filename)
        out = Path(local_dir) / remote_path
        if out.exists() and not force:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        query = urllib.parse.urlencode({"revision": revision})
        url_path = f"/api/repos/{_quote_repo_id(repo_id)}/resolve/{_quote_path(remote_path)}?{query}"
        body = self._request_bytes("GET", url_path)
        tmp = out.with_name(f".{out.name}.tmp")
        tmp.write_bytes(body)
        tmp.replace(out)
        return out

    def snapshot_download(
        self,
        repo_id: str,
        *,
        local_dir: str | Path | None = None,
        revision: str = "main",
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        force: bool = False,
        max_workers: int = 1,
    ) -> Path:
        target = Path(local_dir) if local_dir is not None else Path(repo_id.split("/")[-1])
        files = self.iter_snapshot_files(repo_id, revision=revision, include=include, exclude=exclude)
        self.download_files(
            repo_id,
            [info.path for info in files],
            local_dir=target,
            revision=revision,
            force=force,
            max_workers=max_workers,
        )
        return target

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Mapping[str, Any]] = None,
        data: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        auth: bool = False,
    ) -> Mapping[str, Any]:
        body = data
        req_headers = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        raw = self._request(method, path, data=body, headers=req_headers, auth=auth)
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise MegaHubError("expected JSON object response")
        return parsed

    def _request_bytes(self, method: str, path: str) -> bytes:
        return self._request(method, path)

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        auth: bool = False,
    ) -> bytes:
        req_headers = {"Accept": "application/json", **dict(headers or {})}
        if auth or self.token:
            if not self.token:
                raise MegaHubError("not logged in; run `mega auth login --token ...` first")
            req_headers["Authorization"] = f"Bearer {self.token}"
        if isinstance(data, bytes) and "Content-Length" not in req_headers:
            req_headers["Content-Length"] = str(len(data))
        url = urllib.parse.urlsplit(self.endpoint + path)
        target = url.path or "/"
        if url.query:
            target += f"?{url.query}"
        conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(url.netloc, timeout=120)
        try:
            conn.putrequest(method, target)
            for key, value in req_headers.items():
                conn.putheader(key, value)
            conn.endheaders()
            if data is not None:
                if hasattr(data, "read"):
                    for chunk in iter(lambda: data.read(1024 * 1024), b""):
                        conn.send(chunk)
                elif isinstance(data, bytes):
                    conn.send(data)
                else:
                    conn.send(bytes(data))
            response = conn.getresponse()
            raw = response.read()
            if response.status >= 400:
                message = response.reason
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    message = payload.get("error") or payload.get("message") or message
                except Exception:
                    pass
                raise MegaHubError(
                    f"{response.status} {message}",
                    status_code=response.status,
                    method=method,
                    url=self.endpoint + path,
                )
            return raw
        except OSError as exc:
            raise MegaHubError(str(exc), method=method, url=self.endpoint + path) from exc
        finally:
            conn.close()


def _discussion_page(
    data: Mapping[str, Any],
    *,
    repo_id: str,
    repo_type: str,
    endpoint: str,
) -> DiscussionPage:
    raw_discussions = data.get("discussions", [])
    discussions = tuple(
        _discussion_info(item, repo_id=repo_id, repo_type=repo_type, endpoint=endpoint)
        for item in raw_discussions
        if isinstance(item, Mapping)
    )
    raw_counts = data.get("counts")
    counts = {
        str(key): int(value)
        for key, value in (raw_counts.items() if isinstance(raw_counts, Mapping) else ())
    }
    raw_filter = data.get("filter")
    filters = raw_filter if isinstance(raw_filter, Mapping) else {}
    return DiscussionPage(
        discussions=discussions,
        counts=counts,
        status=str(filters.get("status", "all")),
        kind=str(filters.get("kind", "all")),
        sort=str(filters.get("sort", "recently-created")),
        query=str(filters.get("query", "")),
        page=int(data.get("page", 1)),
        limit=int(data.get("limit", len(discussions) or 20)),
        has_more=bool(data.get("has_more", False)),
    )


def _discussion_thread(
    data: Mapping[str, Any],
    *,
    repo_id: str,
    repo_type: str,
    endpoint: str,
) -> DiscussionThread:
    raw_discussion = data.get("discussion")
    if not isinstance(raw_discussion, Mapping):
        raise MegaHubError("expected discussion response")
    raw_messages = data.get("messages", [])
    raw_permissions = data.get("permissions")
    permissions = raw_permissions if isinstance(raw_permissions, Mapping) else {}
    return DiscussionThread(
        discussion=_discussion_info(
            raw_discussion,
            repo_id=repo_id,
            repo_type=repo_type,
            endpoint=endpoint,
        ),
        messages=tuple(
            _discussion_message(item)
            for item in raw_messages
            if isinstance(item, Mapping)
        ),
        permissions=DiscussionPermissions(
            can_reply=bool(permissions.get("can_reply", False)),
            can_close=bool(permissions.get("can_close", False)),
            can_reopen=bool(permissions.get("can_reopen", False)),
            can_delete=bool(permissions.get("can_delete", False)),
            can_merge=bool(permissions.get("can_merge", False)),
            merge_blocked_reason=(
                str(permissions["merge_blocked_reason"])
                if permissions.get("merge_blocked_reason") is not None
                else None
            ),
        ),
    )


def _discussion_info(
    data: Mapping[str, Any],
    *,
    repo_id: str,
    repo_type: str,
    endpoint: str,
) -> DiscussionInfo:
    raw_pull_request = data.get("pull_request")
    pull_request = None
    if isinstance(raw_pull_request, Mapping):
        pull_request = PullRequestInfo(
            source_branch=str(raw_pull_request.get("source_branch", "")),
            target_branch=str(raw_pull_request.get("target_branch", "")),
            source_revision=str(raw_pull_request.get("source_revision", "")),
            target_revision=str(raw_pull_request.get("target_revision", "")),
            merged_at=(
                str(raw_pull_request["merged_at"])
                if raw_pull_request.get("merged_at") is not None
                else None
            ),
            merged_by=(
                str(raw_pull_request["merged_by"])
                if raw_pull_request.get("merged_by") is not None
                else None
            ),
        )
    raw_author = data.get("author")
    return DiscussionInfo(
        repo_id=repo_id,
        repo_type=repo_type,
        number=int(data.get("number", 0)),
        title=str(data.get("title", "")),
        kind=str(data.get("kind", "discussion")),
        status=(
            "merged"
            if pull_request is not None and pull_request.merged_at is not None
            else str(data.get("status", "open"))
        ),
        author=_community_author(raw_author if isinstance(raw_author, Mapping) else {}),
        reply_count=int(data.get("reply_count", 0)),
        reaction_count=int(data.get("reaction_count", 0)),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        endpoint=endpoint,
        pull_request=pull_request,
    )


def _discussion_message(data: Mapping[str, Any]) -> DiscussionMessageInfo:
    raw_author = data.get("author")
    raw_reaction = data.get("reaction")
    raw_permissions = data.get("permissions")
    reaction = raw_reaction if isinstance(raw_reaction, Mapping) else {}
    permissions = raw_permissions if isinstance(raw_permissions, Mapping) else {}
    return DiscussionMessageInfo(
        message_id=str(data.get("message_id", "")),
        author=_community_author(raw_author if isinstance(raw_author, Mapping) else {}),
        body=str(data.get("body", "")),
        is_original=bool(data.get("is_original", False)),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        reaction_count=int(reaction.get("count", 0)),
        viewer_reacted=bool(reaction.get("viewer_reacted", False)),
        can_edit=bool(permissions.get("can_edit", False)),
        can_delete=bool(permissions.get("can_delete", False)),
    )


def _community_author(data: Mapping[str, Any]) -> CommunityAuthor:
    handle = str(data.get("handle", "deleted"))
    return CommunityAuthor(
        handle=handle,
        display_name=str(data.get("display_name", handle)),
        avatar_url=str(data["avatar_url"]) if data.get("avatar_url") is not None else None,
    )


def _validate_discussion_number(number: int) -> None:
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise ValueError("discussion number must be a positive integer")


def _validate_repo_type(repo_type: str) -> None:
    if repo_type not in {"model", "dataset", "space"}:
        raise ValueError("repo_type must be 'model', 'dataset', or 'space'")


def _repo_info(data: Mapping[str, Any]) -> RepoInfo:
    return RepoInfo(
        repo_id=str(data["repo_id"]),
        private=bool(data.get("private", False)),
        owner=str(data.get("owner", "")),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        repo_type=str(data.get("repo_type", "model")),
        description=str(data.get("description", "")),
        tags=tuple(str(tag) for tag in data.get("tags", []) or []),
        license=str(data.get("license", "")),
        likes=int(data.get("likes", 0)),
        downloads=int(data.get("downloads", 0)),
    )


def _collection_info(data: Mapping[str, Any], *, endpoint: str) -> CollectionInfo:
    raw_owner = data.get("owner")
    if not isinstance(raw_owner, Mapping):
        raise MegaHubError("expected collection owner")
    owner = CollectionOwnerInfo(
        handle=str(raw_owner.get("handle", "")),
        display_name=str(raw_owner.get("display_name", raw_owner.get("handle", ""))),
        kind=str(raw_owner.get("kind", "user")),
    )
    items = tuple(
        CollectionItemInfo(
            item_object_id=str(item["item_object_id"]),
            item_id=str(item["item_id"]),
            item_type=str(item["item_type"]),
            title=str(item.get("title", item["item_id"])),
            description=str(item.get("description", "")),
            href=str(item.get("href", "")),
            note=str(item.get("note", "")),
            position=int(item.get("position", 0)),
            created_at=str(item.get("created_at", "")),
            updated_at=str(item.get("updated_at", "")),
        )
        for item in data.get("items", []) or []
        if isinstance(item, Mapping)
    )
    slug = str(data["slug"])
    return CollectionInfo(
        id=str(data["id"]),
        slug=slug,
        title=str(data["title"]),
        description=str(data.get("description", "")),
        private=bool(data.get("private", False)),
        theme=str(data.get("theme", "neutral")),
        position=int(data.get("position", 0)),
        owner=owner,
        item_count=int(data.get("item_count", len(items))),
        upvotes=int(data.get("upvotes", 0)),
        upvoted=bool(data.get("upvoted", False)),
        items=items,
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        url=f"{endpoint}/collections/{'/'.join(urllib.parse.quote(part, safe='') for part in slug.split('/'))}",
    )


def _account_key_info(data: Mapping[str, Any]) -> AccountKeyInfo:
    return AccountKeyInfo(
        key_id=str(data["key_id"]),
        key_type=str(data["key_type"]),
        name=str(data["name"]),
        public_key=str(data["public_key"]),
        fingerprint=str(data["fingerprint"]),
        created_at=str(data["created_at"]),
        revoked_at=str(data["revoked_at"]) if data.get("revoked_at") is not None else None,
    )


def _webhook_delivery_info(data: Mapping[str, Any]) -> WebhookDeliveryInfo:
    return WebhookDeliveryInfo(
        delivery_id=str(data["delivery_id"]),
        webhook_id=str(data["webhook_id"]),
        event=str(data["event"]),
        state=str(data["state"]),
        attempt_count=int(data.get("attempt_count", 0)),
        response_status=int(data["response_status"]) if data.get("response_status") is not None else None,
        error=str(data["error"]) if data.get("error") is not None else None,
        created_at=str(data.get("created_at", "")),
        delivered_at=str(data["delivered_at"]) if data.get("delivered_at") is not None else None,
    )


def _webhook_last_delivery_info(data: Mapping[str, Any]) -> WebhookLastDeliveryInfo:
    return WebhookLastDeliveryInfo(
        state=str(data["state"]),
        response_status=int(data["response_status"]) if data.get("response_status") is not None else None,
        error=str(data["error"]) if data.get("error") is not None else None,
        delivered_at=str(data.get("delivered_at", "")),
    )


def _webhook_info(data: Mapping[str, Any]) -> WebhookInfo:
    last_delivery_data = data.get("last_delivery")
    return WebhookInfo(
        webhook_id=str(data["webhook_id"]),
        name=str(data["name"]),
        url=str(data["url"]),
        scope=str(data["scope"]),
        repo_id=str(data["repo_id"]) if data.get("repo_id") is not None else None,
        events=tuple(str(event) for event in data.get("events", []) or []),
        enabled=bool(data.get("enabled", False)),
        secret_configured=bool(data.get("secret_configured", False)),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
        last_delivery=_webhook_last_delivery_info(last_delivery_data)
        if isinstance(last_delivery_data, Mapping)
        else None,
    )


def _reject_private_key(value: str) -> None:
    upper = value.upper()
    private_markers = (
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN DSA PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----",
    )
    if any(marker in upper for marker in private_markers):
        raise ValueError("Private keys are never accepted; provide only an SSH .pub file or an armored GPG public key")


def _file_info(data: Mapping[str, Any]) -> FileInfo:
    return FileInfo(
        path=str(data["path"]),
        size=int(data.get("size", 0)),
        sha256=str(data.get("sha256", "")),
        content_type=data.get("content_type"),
    )


def _ref_info(data: Mapping[str, Any]) -> RefInfo:
    return RefInfo(
        name=str(data["name"]),
        ref=str(data.get("ref", "")),
        target_revision=str(data["target_revision"]),
        message=None if data.get("message") is None else str(data["message"]),
        updated_at=str(data.get("updated_at", "")),
    )


def _normalize_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def _quote_repo_id(repo_id: str) -> str:
    namespace, name = _split_repo_id(repo_id)
    return f"{urllib.parse.quote(namespace, safe='')}/{urllib.parse.quote(name, safe='')}"


def _quote_collection_slug(collection_slug: str) -> str:
    owner, slug = _split_repo_id(collection_slug)
    return f"{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(slug, safe='')}"


def _split_repo_id(repo_id: str) -> tuple[str, str]:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repo_id must use the namespace/name form")
    return parts[0], parts[1]


def _quote_path(path: str) -> str:
    return urllib.parse.quote(_clean_repo_path(path), safe="/")


def _clean_repo_path(path: str) -> str:
    cleaned = Path(path).as_posix().lstrip("/")
    if not cleaned or cleaned == "." or ".." in cleaned.split("/"):
        raise ValueError(f"invalid repository path: {path!r}")
    return cleaned


def _clean_repo_copy_path(path: str) -> str:
    """Normalize an optional path used by the repository copy endpoint."""
    if not path:
        return ""
    return _clean_repo_path(path)


def _join_repo_path(prefix: str, path: str) -> str:
    if not prefix:
        return _clean_repo_path(path)
    return _clean_repo_path(f"{prefix.rstrip('/')}/{path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_state_path(
    repo_id: str,
    path: str,
    revision: str,
    size: int,
    sha256: str,
    *,
    stage_only: bool = False,
) -> Path:
    identity = hashlib.sha256(
        f"{repo_id}\n{path}\n{revision}\n{size}\n{sha256}\n{int(stage_only)}".encode("utf-8")
    ).hexdigest()
    return config_dir() / "uploads" / f"{identity}.json"


class _FileSlice:
    """Bound a seeked file object to one multipart request body."""

    def __init__(self, file: Any, remaining: int) -> None:
        self._file = file
        self._remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._file.read(size)
        self._remaining -= len(data)
        return data


def _load_upload_state(path: Path) -> Mapping[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _save_upload_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, sort_keys=True)
        file.write("\n")
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)


def _matches_upload_state(
    state: Mapping[str, Any] | None,
    endpoint: str,
    repo_id: str,
    path: str,
    revision: str,
    size: int,
    sha256: str,
    *,
    stage_only: bool = False,
) -> bool:
    return bool(
        state
        and state.get("endpoint") == endpoint
        and state.get("repo_id") == repo_id
        and state.get("path") == path
        and state.get("revision") == revision
        and state.get("size") == size
        and state.get("sha256") == sha256
        and state.get("stage_only", False) is stage_only
        and isinstance(state.get("upload_id"), str)
        and isinstance(state.get("part_size"), int)
    )


def _iter_upload_files(
    root: Path,
    *,
    include: Optional[Iterable[str]],
    exclude: Optional[Iterable[str]],
) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _matches_patterns(rel, include=include, exclude=exclude):
            yield path


def _matches_patterns(
    path: str,
    *,
    include: Optional[Iterable[str]],
    exclude: Optional[Iterable[str]],
) -> bool:
    include_patterns = list(include or [])
    exclude_patterns = list(exclude or [])
    if include_patterns and not any(fnmatch.fnmatch(path, pattern) for pattern in include_patterns):
        return False
    if exclude_patterns and any(fnmatch.fnmatch(path, pattern) for pattern in exclude_patterns):
        return False
    return True

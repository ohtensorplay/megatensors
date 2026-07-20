# Copyright 2026-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Centralized parser for MEGA Hub URIs ('mega://...') and mount specifications.

A MEGA URI is a URI-like string that identifies a location on the MEGA
Hub: a model/dataset/space/kernel repository, a bucket, optionally a revision,
and optionally a path inside the repo or bucket.

Canonical syntax:

```
mega://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
```

For convenience, [`parse_mega_uri`] also accepts MEGA **web URLs** (the
ones you copy-paste from your browser), e.g.
'https://mega.tensorplay.cn/datasets/my-org/my-dataset/blob/main/train.csv'. They are
normalized to the canonical 'mega://' form before parsing. Only unambiguous URLs
(repository / bucket pages and file/folder viewer routes) are accepted; any other
route is rejected rather than guessed.

A MEGA mount wraps a MEGA URI with a local mount path and an optional ':ro'/':rw'
flag (used by Spaces and Jobs volumes):

```
mega://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
```

See the MEGA Hub quickstart for the full grammar and examples.
"""

import functools
import re
from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlsplit

from megatensors._hub import constants
from megatensors._hub.errors import MegaUriError, MegaValidationError

from ._validators import validate_repo_id


# Inverse map (singular -> plural URI prefix). Built once from the canonical
# 'constants.MEGA_URI_TYPE_PREFIXES' and used to render URIs.
_TYPE_TO_PREFIX: dict[str, str] = {v: k for k, v in constants.MEGA_URI_TYPE_PREFIXES.items()}

# Special revisions that contain a '/'. They take precedence when splitting
# the part after '@' into '<revision>/<path-in-repo>'. Matches 'refs/pr/N'
# (Pull Request refs) and 'refs/convert/<name>' (e.g. parquet conversions).
# The conversion name allows the typical git ref characters '[a-zA-Z0-9_.-]'
# so names like 'parquet-v2' or 'duckdb.v1' round-trip correctly.
_SPECIAL_REFS_REVISION_REGEX = re.compile(r"^refs/(?:convert/[\w.-]+|pr/\d+)")

# Same as constants.MegaUriType, but as a set of strings for easy lookup.)
_VALID_URI_TYPES: frozenset[str] = frozenset(constants.MEGA_URI_TYPE_PREFIXES.values())


# Web-viewer routes that point at a file or folder and that map cleanly onto a
# '<revision>/<path>' pair. Other routes (commit, commits, discussions, settings,
# edit, ...) do not identify a Hub location and are rejected by the URL parser.
_URL_REPO_LOCATION_ACTIONS: frozenset[str] = frozenset({"blame", "blob", "raw", "resolve", "tree"})
# Bucket web routes that point at a file or folder. Buckets are not versioned, so
# these are followed directly by '<path>' (no revision segment).
_URL_BUCKET_LOCATION_ACTIONS: frozenset[str] = frozenset({"resolve", "tree"})


@dataclass(frozen=True)
class MegaUri:
    """Parsed representation of a MEGA Hub URI ('mega://...').

    Attributes:
        type (`str`):
            One of 'model', 'dataset', 'space', 'kernel' or 'bucket'.
        id (`str`):
            The repository id ('namespace/name', e.g. 'my-org/my-model') for repo URIs, or the bucket id ('namespace/name') for bucket URIs.
        revision (`str`, *optional*):
            The revision specified after '@' in the URI, URL-decoded. 'None' if no revision was specified, or for bucket URIs (which
            never carry a revision). Special refs like 'refs/pr/10' and 'refs/convert/parquet' are preserved as-is.
        path_in_repo (`str`):
            The path inside the repo or bucket. Empty string if the URI points at the root.
    """

    type: constants.MegaUriType
    id: str
    revision: str | None = None
    path_in_repo: str = ""
    _raw: str | None = field(repr=False, hash=False, compare=False, default=None)

    def __post_init__(self) -> None:
        uri = self._raw or ""  # For error messages

        # Check valid URI type
        if self.type not in _VALID_URI_TYPES:
            raise MegaUriError(uri=uri, msg=f"Invalid type '{self.type}'. Must be one of {sorted(_VALID_URI_TYPES)}.")

        # Check valid ID
        if not self.id or self.id.count("/") != 1:
            raise MegaUriError(uri=uri, msg=f"Id must be 'namespace/name', got '{self.id}'.")
        if self.type != "bucket":
            try:
                validate_repo_id(self.id)
            except MegaValidationError as e:
                raise MegaUriError(uri=uri, msg=str(e)) from e

        # Check valid revision
        if self.revision is not None and not self.revision:
            raise MegaUriError(uri=uri, msg="Revision must not be an empty string.")
        if self.type == "bucket" and self.revision is not None:
            raise MegaUriError(uri=uri, msg="Bucket URIs do not support a revision.")

        # Check valid path in repo
        if self.path_in_repo:
            if self.path_in_repo.startswith("/") or "//" in self.path_in_repo:
                raise MegaUriError(uri=uri, msg=f"Path must not contain empty segments (got '{self.path_in_repo}').")

    @property
    def is_bucket(self) -> bool:
        """True if this URI points at a bucket."""
        return self.type == "bucket"

    @property
    def is_repo(self) -> bool:
        """True if this URI points at a repository (model, dataset, space or kernel)."""
        return self.type != "bucket"

    def to_uri(self) -> str:
        """Render the URI as a canonical 'mega://' string.

        The type prefix is always written explicitly (e.g. 'mega://models/my-org/my-model').
        """
        parts: list[str] = [constants.MEGA_PROTOCOL, _TYPE_TO_PREFIX[self.type], "/", self.id]
        if self.revision is not None:
            # Encode '/' as '%2F' for revisions that would otherwise be split as '<revision>/<path>'
            # at parse time. Special refs ('refs/pr/N', 'refs/convert/<name>') are kept verbatim
            # because the parser matches them eagerly.
            revision = self.revision
            if "/" in revision and _SPECIAL_REFS_REVISION_REGEX.fullmatch(revision) is None:
                revision = revision.replace("/", "%2F")
            parts.append(f"@{revision}")
        if self.path_in_repo:
            parts.append(f"/{self.path_in_repo}")
        return "".join(parts)

    def to_url(self, endpoint: str | None = None) -> str:
        """Render the URI as a MEGA **web URL** (the kind you open in a browser).

        This is the inverse of parsing a URL with [`parse_mega_uri`]. The returned URL points at:

        - the repository / bucket landing page when no path or revision is set;
        - the folder viewer ('/tree/<revision>') when only a revision is set;
        - the file viewer ('/blob/<revision>/<path>') for repository files (revision defaults to 'main');
        - the tree route ('/tree/<path>') for bucket files (buckets are not versioned).

        Args:
            endpoint (`str`, *optional*):
                Base endpoint to use. Defaults to 'constants.ENDPOINT' (i.e. 'https://mega.tensorplay.cn').

        Returns:
            `str`: the web URL.

        Example:
            ```py
            >>> from megatensors._hub import parse_mega_uri
            >>> parse_mega_uri("mega://datasets/my-org/my-dataset@v1/train.csv").to_url()
            'https://mega.tensorplay.cn/datasets/my-org/my-dataset/blob/v1/train.csv'
            ```
        """
        base = (endpoint or constants.ENDPOINT).rstrip("/")
        # Percent-encode characters that would otherwise break the URL (spaces, '#', '?', ...),
        # keeping '/' as the path separator. This is the inverse of the decoding done when parsing.
        path = quote(self.path_in_repo, safe="/")

        if self.type == "bucket":
            url = f"{base}/buckets/{self.id}"
            if path:
                url += f"/tree/{path}"
            return url

        # Models live at the root ('MEGA Hub/<id>'); other repos are namespaced by their plural prefix.
        url = f"{base}/{self.id}" if self.type == "model" else f"{base}/{_TYPE_TO_PREFIX[self.type]}/{self.id}"
        revision = self.revision
        # Percent-encode the branch/tag name so it stays a single, URL-safe segment: a '/' would
        # otherwise open a new path segment and '#'/'?' would be read as a fragment/query when the
        # URL is opened or parsed back. This mirrors the decoding done when parsing. Special refs
        # ('refs/pr/N', 'refs/convert/<name>') are used verbatim by the Hub web routes.
        if revision is not None and _SPECIAL_REFS_REVISION_REGEX.fullmatch(revision) is None:
            revision = quote(revision, safe="")
        if path:
            url += f"/blob/{revision or constants.DEFAULT_REVISION}/{path}"
        elif revision is not None:
            url += f"/tree/{revision}"
        return url


@dataclass(frozen=True)
class MegaMount:
    """A MEGA URI paired with a local mount path and optional read-only flag.

    Used by Spaces and Jobs to describe volume mounts. The full syntax is:

    ```
    mega://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
    ```

    Attributes:
        source ([`MegaUri`]):
            The parsed MEGA URI identifying the Hub resource to mount.
        mount_path (`str`):
            The local mount path (always starts with '/').
        read_only (`bool`, *optional*):
            True if the mount ends with ':ro', False if it ends with ':rw', 'None' if no flag was provided.
    """

    source: MegaUri
    mount_path: str
    read_only: bool | None = None
    _raw: str | None = field(repr=False, hash=False, compare=False, default=None)

    def __post_init__(self) -> None:
        raw = self._raw or ""
        if not self.mount_path.startswith("/") or self.mount_path == "/":
            raise MegaUriError(
                uri=raw,
                msg=f"Mount path must be a non-empty absolute path starting with '/', got '{self.mount_path}'.",
            )

    def to_uri(self) -> str:
        """Render the mount as a canonical 'mega://' string.

        Example: 'mega://models/my-org/my-model:/data:ro'
        """
        parts = [self.source.to_uri(), ":", self.mount_path]
        if self.read_only is not None:
            parts.append(":ro" if self.read_only else ":rw")
        return "".join(parts)


def is_mega_uri(uri: str) -> bool:
    """Check if a string is a valid MEGA URI ('mega://...') or a recognized MEGA web URL."""
    try:
        parse_mega_uri(uri)
        return True
    except MegaUriError:
        return False


@functools.lru_cache
def parse_mega_uri(uri: str, endpoint: str | None = None) -> MegaUri:
    """Parse a MEGA Hub URI ('mega://...') or a MEGA web URL.

    A MEGA URI is a URI-like string identifying a location on the MEGA Hub. The full grammar is:

    ```
    mega://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
    ```

    For convenience, MEGA **web URLs** (the ones you copy-paste from the website) are also
    accepted and normalized to the canonical 'mega://' form, e.g.
    'https://mega.tensorplay.cn/datasets/my-org/my-dataset/blob/main/train.csv'. Only unambiguous URLs
    (repository / bucket pages and file/folder viewer routes) are accepted; any other route is rejected.

    See the MEGA Hub quickstart for the full specification.

    Args:
        uri (`str`):
            The URI to parse. Must start with 'mega://', or be a MEGA URL (e.g. 'https://mega.tensorplay.cn/...').
        endpoint (`str`, *optional*):
            A custom Hub endpoint (e.g. a self-hosted or proxied Hub like 'https://hub.my-company.com' or
            'http://localhost:8080/mega'). When provided, web URLs on that endpoint are recognized in addition to
            the default MEGA hosts. Has no effect on 'mega://' URIs.

    Returns:
        [`MegaUri`]: the parsed URI.

    Raises:
        [`MegaUriError`]:
            If the URI is malformed (missing prefix, invalid type, missing id, unsupported URL route, etc.).

    Examples:
        ```py
        >>> from megatensors._hub.utils import parse_mega_uri
        >>> parse_mega_uri("mega://my-org/my-model")
        MegaUri(type='model', id='my-org/my-model', revision=None, path_in_repo='')
        >>> parse_mega_uri("mega://datasets/my-org/my-dataset@refs/pr/3/train.json")
        MegaUri(type='dataset', id='my-org/my-dataset', revision='refs/pr/3', path_in_repo='train.json')
        >>> parse_mega_uri("https://mega.tensorplay.cn/datasets/my-org/my-dataset/blob/main/train.csv")
        MegaUri(type='dataset', id='my-org/my-dataset', revision='main', path_in_repo='train.csv')
        ```
    """
    raw = uri
    if uri.startswith(constants.MEGA_PROTOCOL):
        body = uri[len(constants.MEGA_PROTOCOL) :]
        if not body:
            raise MegaUriError(uri, f"Empty body after '{constants.MEGA_PROTOCOL}'.")
    elif _looks_like_mega_url(uri, endpoint=endpoint):
        body = _url_to_uri_body(uri, endpoint=endpoint)
    else:
        raise MegaUriError(
            uri,
            f"Must start with '{constants.MEGA_PROTOCOL}' or be a MEGA URL (e.g. 'https://mega.tensorplay.cn/...'). "
            f"Expected format: {constants.MEGA_PROTOCOL}[<TYPE>/]<ID>[@<REVISION>][/<PATH>]",
        )

    type_, location = _split_type(body, raw=raw)

    if type_ == "bucket":
        return _parse_bucket_body(location, type_, raw=raw)
    return _parse_repo_body(location, type_, raw=raw)


def _endpoint_host_and_path(endpoint: str | None) -> tuple[str | None, str]:
    """Return the lowercased host and stripped path prefix of a custom Hub 'endpoint'.

    E.g. 'https://hub.my-company.com' -> ('hub.my-company.com', '') and a self-hosted
    'http://localhost:8080/mega' -> ('localhost', 'mega'). Returns '(None, "")' when 'endpoint' is None.
    """
    if endpoint is None:
        return None, ""
    # Prefix '//' for scheme-less endpoints so 'urlsplit' populates 'netloc' instead of 'path'.
    parsed = urlsplit(endpoint if "://" in endpoint else "//" + endpoint)
    host = parsed.hostname.lower() if parsed.hostname else None
    return host, parsed.path.strip("/")


def _recognized_hosts(endpoint: str | None) -> frozenset[str]:
    """The set of hosts whose web URLs can be parsed: the default MEGA hosts plus 'endpoint'."""
    host, _ = _endpoint_host_and_path(endpoint)
    return constants.MEGA_URL_HOSTS | {host} if host else constants.MEGA_URL_HOSTS


def _looks_like_mega_url(uri: str, endpoint: str | None = None) -> bool:
    """Return True if 'uri' looks like a (possibly scheme-less) MEGA web URL."""
    lowered = uri.lower()
    if lowered.startswith(("http://", "https://")):
        return True
    # Scheme-less host (e.g. 'mega.tensorplay.cn/org/model').
    return any(lowered == host or lowered.startswith(host + "/") for host in _recognized_hosts(endpoint))


def _decode_url_path_segment(segment: str) -> str:
    """Percent-decode a single URL path segment (e.g. 'file%20name.txt' -> 'file name.txt').

    A decoded '/' is re-encoded as '%2F' so the segment stays atomic when the normalized body is
    re-split by the shared parser. This decodes ordinary path characters (spaces, '#', ...) that
    browsers encode, while keeping '%2F'-encoded revisions (e.g. 'feature%2Ffoo') intact.
    """
    return unquote(segment).replace("/", "%2F")


def _url_to_uri_body(url: str, endpoint: str | None = None) -> str:
    """Normalize a MEGA web URL into the body of a 'mega://' URI (everything after 'mega://').

    The returned string is fed back into the regular URI parsing logic, so all validation
    (repo id, revision, empty path segments, ...) is shared with the canonical 'mega://' path.
    Only unambiguous URLs are accepted: any unrecognized route raises [`MegaUriError`]. When 'endpoint'
    is provided, URLs on that custom Hub host are recognized too (and its path prefix is stripped).
    """
    raw = url
    # Prefix '//' for scheme-less inputs so 'urlsplit' populates 'netloc' instead of 'path'.
    parsed = urlsplit(url if "://" in url else "//" + url)
    host = (parsed.hostname or "").lower()
    if host not in _recognized_hosts(endpoint):
        raise MegaUriError(
            uri=raw,
            msg=f"Unrecognized host '{host or url}'. Expected a MEGA URL (e.g. 'https://mega.tensorplay.cn/...').",
        )

    # Query string and fragment are intentionally dropped (e.g. '?download=true').
    path = parsed.path
    # For a self-hosted endpoint with a path prefix (e.g. 'http://localhost:8080/mega'), drop it so the
    # remaining segments are '[<TYPE>/]<namespace>/<name>[/...]' just like on the public Hub.
    endpoint_host, endpoint_path = _endpoint_host_and_path(endpoint)
    if endpoint_path and host == endpoint_host:
        prefix = "/" + endpoint_path
        if path == prefix or path.startswith(prefix + "/"):
            path = path[len(prefix) :]
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        raise MegaUriError(uri=raw, msg=f"Missing repository or bucket identifier in URL '{url}'.")

    # Optional type prefix ('datasets', 'spaces', 'kernels', 'buckets', 'models').
    type_prefix: str | None = None
    if segments[0] in constants.MEGA_URI_TYPE_PREFIXES:
        type_prefix = segments[0]
        segments = segments[1:]

    # Everything in the web UI is namespaced ('<namespace>/<name>'); a single segment is a user or
    # organization page (or a listing page), which we cannot map to a repository -> reject.
    if len(segments) < 2:
        raise MegaUriError(
            uri=raw,
            msg=(
                f"Cannot parse URL '{url}': expected a '<namespace>/<name>' repository or bucket. "
                "User/organization pages and single-segment URLs are not supported."
            ),
        )
    repo_id = f"{segments[0]}/{segments[1]}"
    rest = segments[2:]

    if type_prefix == "buckets":
        if not rest:
            return f"buckets/{repo_id}"
        action, *tail = rest
        if action not in _URL_BUCKET_LOCATION_ACTIONS:
            raise MegaUriError(uri=raw, msg=f"Cannot parse bucket URL '{url}': unsupported '/{action}/' route.")
        path = "/".join(_decode_url_path_segment(segment) for segment in tail)
        return f"buckets/{repo_id}/{path}" if path else f"buckets/{repo_id}"

    prefix = f"{type_prefix}/" if type_prefix else ""
    if not rest:
        return f"{prefix}{repo_id}"
    action, *tail = rest
    if action not in _URL_REPO_LOCATION_ACTIONS:
        raise MegaUriError(
            uri=raw,
            msg=(
                f"Cannot parse URL '{url}': unsupported '/{action}/' route. "
                "Only repository pages and file/folder viewer routes (blob, resolve, raw, tree, ...) can be parsed."
            ),
        )
    if not tail:
        # e.g. '.../tree' with nothing after -> repository root.
        return f"{prefix}{repo_id}"
    # 'tail' is '<revision>/<path>'; reuse the canonical '@<revision>/<path>' splitting logic
    # (special refs, URL-encoded slashes, ...) by handing it back to the URI parser. Each segment
    # is percent-decoded first so file names with spaces, '#', ... resolve correctly; the revision
    # segment's '%2F' survives (re-encoded by '_decode_url_path_segment') and is decoded downstream.
    decoded = "/".join(_decode_url_path_segment(segment) for segment in tail)
    return f"{prefix}{repo_id}@{decoded}"


def parse_mega_mount(mount_str: str) -> MegaMount:
    """Parse a MEGA mount specification ('mega://...:<MOUNT_PATH>[:ro|:rw]').

    A mount specification is a MEGA URI followed by a local mount path and an optional read-only/read-write flag.
    The full grammar is:

    ```
    mega://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
    ```

    See the MEGA Hub quickstart for the full specification.

    Args:
        mount_str (`str`):
            The mount string to parse. Must start with 'mega://' and contain a ':<MOUNT_PATH>' segment.

    Returns:
        [`MegaMount`]: the parsed mount.

    Raises:
        [`MegaUriError`]:
            If the mount string is malformed (missing mount path, invalid URI, etc.).

    Examples:
        ```py
        >>> from megatensors._hub.utils import parse_mega_mount
        >>> parse_mega_mount("mega://my-org/my-model:/data:ro")
        MegaMount(source=MegaUri(type='model', id='my-org/my-model', revision=None, path_in_repo=''), mount_path='/data', read_only=True)
        >>> parse_mega_mount("mega://buckets/my-org/my-bucket/sub/dir:/mnt:rw")
        MegaMount(source=MegaUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir'), mount_path='/mnt', read_only=False)
        ```
    """
    if not mount_str.startswith(constants.MEGA_PROTOCOL):
        raise MegaUriError(
            uri=mount_str,
            msg=f"Must start with '{constants.MEGA_PROTOCOL}'.",
        )

    raw = mount_str
    body = mount_str[len(constants.MEGA_PROTOCOL) :]
    if not body:
        raise MegaUriError(uri=raw, msg=f"Empty body after '{constants.MEGA_PROTOCOL}'.")

    location, mount_path, read_only = _split_mount(body, raw=raw)

    if mount_path is None:
        raise MegaUriError(uri=raw, msg="Missing mount path. Expected ':<MOUNT_PATH>' (e.g. 'mega://org/model:/data').")

    # Re-assemble the URI part and parse it
    uri_str = constants.MEGA_PROTOCOL + location
    try:
        source = parse_mega_uri(uri_str)
    except MegaUriError as e:
        raise MegaUriError(uri=raw, msg=e.msg) from e

    return MegaMount(source=source, mount_path=mount_path, read_only=read_only, _raw=raw)


def _split_mount(body: str, *, raw: str) -> tuple[str, str | None, bool | None]:
    """Split the ':<MOUNT_PATH>[:ro|:rw]' suffix from 'body'.

    Returns '(location, mount_path, read_only)' where 'mount_path' is 'None' if no mount segment is present.
    """
    if body.endswith(":ro"):
        read_only, body = True, body.removesuffix(":ro")
    elif body.endswith(":rw"):
        read_only, body = False, body.removesuffix(":rw")
    else:
        read_only = None

    # Mount paths always start with '/', so the delimiter is ':/'.
    # We use rfind() because the mount segment is always trailing
    idx = body.rfind(":/")
    if idx == -1:
        if read_only is not None:
            raise MegaUriError(
                uri=raw,
                msg="':ro'/':rw' suffix is only valid when a mount path is provided (e.g. 'mega://...:/<MOUNT_PATH>:ro').",
            )
        return body, None, None

    location = body[:idx]
    mount_path = body[idx + 1 :]  # includes the leading '/'
    if not location:
        raise MegaUriError(uri=raw, msg="Missing location before mount path.")
    return location, mount_path, read_only


def _split_type(location: str, *, raw: str) -> tuple[constants.MegaUriType, str]:
    """Detect the (optional) type prefix and return '(type, remaining_location)'.

    A missing type prefix defaults to 'model'. Singular forms ('model/', 'dataset/', etc.) are explicitly rejected with a helpful error.
    """
    slash_idx = location.find("/")
    if slash_idx == -1:
        # Single segment, no prefix. Reject if it looks like a bare type name.
        if location in constants.MEGA_URI_TYPE_PREFIXES:
            raise MegaUriError(
                uri=raw,
                msg=f"Missing identifier after '{location}'. Expected '{constants.MEGA_PROTOCOL}{location}/<ID>'.",
            )
        if (singular_plural := _TYPE_TO_PREFIX.get(location)) is not None:
            raise MegaUriError(
                uri=raw,
                msg=f"Type prefix must be plural. Did you mean '{constants.MEGA_PROTOCOL}{singular_plural}/...'?",
            )
        return "model", location

    first = location[:slash_idx]
    rest = location[slash_idx + 1 :]
    if first in constants.MEGA_URI_TYPE_PREFIXES:
        return constants.MEGA_URI_TYPE_PREFIXES[first], rest
    if (singular_plural := _TYPE_TO_PREFIX.get(first)) is not None:
        raise MegaUriError(
            uri=raw, msg=f"Type prefix must be plural, got '{first}/'. Did you mean '{singular_plural}/'?"
        )
    return "model", location


def _parse_bucket_body(
    location: str,
    type_: constants.MegaUriType,
    *,
    raw: str,
) -> MegaUri:
    """Parse the body of a bucket URI: 'namespace/name[/path]'."""
    location = location.strip("/")
    parts = location.split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise MegaUriError(uri=raw, msg=f"Bucket id must be 'namespace/name', got '{location}'.")
    bucket_id = f"{parts[0]}/{parts[1]}"
    if "@" in bucket_id:
        raise MegaUriError(uri=raw, msg="Bucket URIs do not support a revision marker ('@').")
    path_in_bucket = parts[2] if len(parts) >= 3 else ""
    return MegaUri(
        type=type_,
        id=bucket_id,
        revision=None,
        path_in_repo=path_in_bucket,
        _raw=raw,
    )


def _parse_repo_body(
    location: str,
    type_: constants.MegaUriType,
    *,
    raw: str,
) -> MegaUri:
    """Parse the body of a repo URI: '<repo_id>[@<revision>][/<path>]'."""
    location = location.strip("/")
    if not location:
        raise MegaUriError(uri=raw, msg="Missing repository id.")

    # The '@' separates the repo_id from the revision, but only when it
    # appears right after 'namespace/name' (at most one '/' before it).
    # An '@' deeper in the path (e.g. in a filename like 'file@1.txt') is literal.
    at_idx = location.find("@")
    revision: str | None

    if at_idx == -1 or location[:at_idx].count("/") > 1:
        # No '@' at all, or the '@' is past the repo_id portion (in a filename).
        revision = None
        parts = location.split("/", 2)
        if len(parts) < 2:
            raise MegaUriError(uri=raw, msg=f"Repository id must be 'namespace/name', got '{location}'. ")
        repo_id = f"{parts[0]}/{parts[1]}"
        path_in_repo = parts[2] if len(parts) > 2 else ""
    else:
        repo_id = location[:at_idx]
        rev_and_path = location[at_idx + 1 :]
        if not repo_id:
            raise MegaUriError(uri=raw, msg="Missing repository id before '@'.")
        if repo_id.count("/") != 1:
            raise MegaUriError(uri=raw, msg=f"Repository id must be 'namespace/name', got '{repo_id}'.")
        # Special refs like 'refs/pr/10' contain '/' and must be matched eagerly,
        # otherwise we would split them at the first '/' and treat the rest as a path.
        match = _SPECIAL_REFS_REVISION_REGEX.match(rev_and_path)
        if match is not None:
            revision = match.group()
            path_in_repo = rev_and_path[len(revision) :].removeprefix("/")
        else:
            slash_idx = rev_and_path.find("/")
            if slash_idx == -1:
                revision = rev_and_path
                path_in_repo = ""
            else:
                revision = rev_and_path[:slash_idx]
                path_in_repo = rev_and_path[slash_idx + 1 :]
        revision = unquote(revision)
        if not revision:
            raise MegaUriError(uri=raw, msg="Empty revision after '@'.")

    return MegaUri(
        type=type_,
        id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        _raw=raw,
    )


# Preserve the established Hub import names while parsing MEGA URIs and web
# URLs. ``parse_hf_uri`` is intentionally not a parser for Hugging Face URLs.
HfUri = MegaUri
parse_hf_uri = parse_mega_uri

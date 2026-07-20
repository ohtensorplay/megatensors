# Copyright 2025-present, the HuggingFace Inc. team.
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
"""Contains the `mega cache` command group and its cache management subcommands."""

import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import click

from megatensors._hub import constants
from megatensors._hub.errors import CLIError
from megatensors._hub.file_download import repo_folder_name
from megatensors.hub import MegaHubClient

from ..utils import (
    ANSI,
    CachedRepoInfo,
    CachedRevisionInfo,
    CacheNotFound,
    MegaCacheInfo as CacheInfo,
    _format_size,
    scan_cache_dir,
)
from ..utils._parsing import parse_duration, parse_size
from ._cli_utils import RepoIdArg, RepoTypeOpt, RevisionOpt, TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out


cache_cli = typer_factory(help="Manage the local MEGA cache directory.")

#### Cache helper utilities


@dataclass(frozen=True)
class _DeletionResolution:
    revisions: frozenset[str]
    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]]
    missing: tuple[str, ...]


_FILTER_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)\s*(?P<op>==|!=|>=|<=|>|<|=)\s*(?P<value>.+)$")
_ALLOWED_OPERATORS = {"=", "!=", ">", "<", ">=", "<="}
_FILTER_KEYS = {"accessed", "modified", "refs", "size", "type"}
_SORT_KEYS = {"accessed", "modified", "name", "size"}
_SORT_PATTERN = re.compile(r"^(?P<key>[a-zA-Z_]+)(?::(?P<order>asc|desc))?$")
_SORT_DEFAULT_ORDER = {
    # Default ordering: accessed/modified/size are descending (newest/biggest first), name is ascending
    "accessed": "desc",
    "modified": "desc",
    "size": "desc",
    "name": "asc",
}


# Dynamically generate SortOptions enum from _SORT_KEYS
_sort_options_dict = {}
for key in sorted(_SORT_KEYS):
    _sort_options_dict[key] = key
    _sort_options_dict[f"{key}_asc"] = f"{key}:asc"
    _sort_options_dict[f"{key}_desc"] = f"{key}:desc"

SortOptions = Enum("SortOptions", _sort_options_dict, type=str, module=__name__)  # type: ignore


@dataclass(frozen=True)
class CacheDeletionCounts:
    """Simple counters summarizing cache deletions for CLI messaging."""

    repo_count: int
    partial_revision_count: int
    total_revision_count: int


CacheEntry = tuple[CachedRepoInfo, CachedRevisionInfo | None]
RepoRefsMap = dict[CachedRepoInfo, frozenset[str]]


def summarize_deletions(
    selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]],
) -> CacheDeletionCounts:
    """Summarize deletions across repositories."""
    repo_count = 0
    total_revisions = 0
    revisions_in_full_repos = 0

    for repo, revisions in selected_by_repo.items():
        total_revisions += len(revisions)
        if len(revisions) == len(repo.revisions):
            repo_count += 1
            revisions_in_full_repos += len(revisions)

    partial_revision_count = total_revisions - revisions_in_full_repos
    return CacheDeletionCounts(repo_count, partial_revision_count, total_revisions)


def _prune_summary(revision_count: int, incomplete_count: int) -> str:
    """Build the human-readable summary of what `mega cache prune` is about to delete."""
    parts: list[str] = []
    if revision_count:
        parts.append(f"{revision_count} unreferenced revision(s)")
    if incomplete_count:
        parts.append(f"{incomplete_count} incomplete download(s)")
    return " and ".join(parts)


def print_cache_selected_revisions(selected_by_repo: Mapping[CachedRepoInfo, frozenset[CachedRevisionInfo]]) -> None:
    """Pretty-print selected cache revisions during confirmation prompts."""
    for repo in sorted(selected_by_repo.keys(), key=lambda repo: (repo.repo_type, repo.repo_id.lower())):
        repo_key = f"{repo.repo_type}/{repo.repo_id}"
        revisions = sorted(selected_by_repo[repo], key=lambda rev: rev.commit_hash)
        if len(revisions) == len(repo.revisions):
            out.text(f"  - {repo_key} (entire repo)")
            continue

        out.text(f"  - {repo_key}:")
        for revision in revisions:
            refs = " ".join(sorted(revision.refs)) or "(detached)"
            out.text(f"      {revision.commit_hash} [{refs}] {revision.size_on_disk_str}")


def build_cache_index(
    cache_info: CacheInfo,
) -> tuple[
    dict[str, CachedRepoInfo],
    dict[str, tuple[CachedRepoInfo, CachedRevisionInfo]],
]:
    """Create lookup tables so CLI commands can resolve repo ids and revisions quickly."""
    repo_lookup: dict[str, CachedRepoInfo] = {}
    revision_lookup: dict[str, tuple[CachedRepoInfo, CachedRevisionInfo]] = {}
    for repo in cache_info.repos:
        repo_key = repo.cache_id.lower()
        repo_lookup[repo_key] = repo
        for revision in repo.revisions:
            revision_lookup[revision.commit_hash.lower()] = (repo, revision)
    return repo_lookup, revision_lookup


def _repo_cache_id_from_target(target: str) -> str:
    """Return the cache id matching a repository target passed to `mega cache rm`."""
    if not target.startswith("mega://"):
        if "://" in target:
            raise CLIError("Repository URIs must use the `mega://` scheme.")
        return target

    location = target.removeprefix("mega://").strip("/")
    if "@" in location:
        raise CLIError("Only repo-level `mega://` URIs are supported by `mega cache rm`.")
    parts = location.split("/")
    if len(parts) == 2:
        return f"model/{location}"
    if len(parts) == 3 and parts[0] in {"models", "datasets", "spaces"}:
        return f"{parts[0][:-1]}/{parts[1]}/{parts[2]}"
    raise CLIError(
        "Invalid MEGA repository URI. Expected `mega://namespace/repository` or "
        "`mega://[models|datasets|spaces]/namespace/repository`."
    )


def collect_cache_entries(
    cache_info: CacheInfo, *, include_revisions: bool
) -> tuple[list[CacheEntry], RepoRefsMap]:
    """Flatten cache metadata into rows consumed by `mega cache ls`."""
    entries: list[CacheEntry] = []
    repo_refs_map: RepoRefsMap = {}
    sorted_repos = sorted(cache_info.repos, key=lambda repo: (repo.repo_type, repo.repo_id.lower()))
    for repo in sorted_repos:
        repo_refs_map[repo] = frozenset({ref for revision in repo.revisions for ref in revision.refs})
        if include_revisions:
            for revision in sorted(repo.revisions, key=lambda rev: rev.commit_hash):
                entries.append((repo, revision))
        else:
            entries.append((repo, None))
    if include_revisions:
        entries.sort(
            key=lambda entry: (
                entry[0].cache_id,
                entry[1].commit_hash if entry[1] is not None else "",
            )
        )
    else:
        entries.sort(key=lambda entry: entry[0].cache_id)
    return entries, repo_refs_map


def compile_cache_filter(
    expr: str, repo_refs_map: RepoRefsMap
) -> Callable[[CachedRepoInfo, CachedRevisionInfo | None, float], bool]:
    """Convert a `mega cache ls` filter expression into the predicate for each cache entry."""
    match = _FILTER_PATTERN.match(expr.strip())
    if not match:
        raise ValueError(f"Invalid filter expression: '{expr}'.")

    key = match.group("key").lower()
    op = match.group("op")
    value_raw = match.group("value").strip()

    if op not in _ALLOWED_OPERATORS:
        raise ValueError(f"Unsupported operator '{op}' in filter '{expr}'. Must be one of {list(_ALLOWED_OPERATORS)}.")

    if key not in _FILTER_KEYS:
        raise ValueError(f"Unsupported filter key '{key}' in '{expr}'. Must be one of {list(_FILTER_KEYS)}.")
    # at this point we know that key is in `_FILTER_KEYS`
    if key == "size":
        size_threshold = parse_size(value_raw)
        return lambda repo, revision, _: _compare_numeric(
            revision.size_on_disk if revision is not None else repo.size_on_disk,
            op,
            size_threshold,
        )

    if key in {"modified", "accessed"}:
        seconds = parse_duration(value_raw.strip())

        def _time_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, now: float) -> bool:
            timestamp = (
                repo.last_accessed
                if key == "accessed"
                else revision.last_modified
                if revision is not None
                else repo.last_modified
            )
            if timestamp is None:
                return False
            return _compare_numeric(now - timestamp, op, seconds)

        return _time_filter

    if key == "type":
        expected = value_raw.lower()

        if op != "=":
            raise ValueError(f"Only '=' is supported for 'type' filters. Got '{op}'.")

        def _type_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, _: float) -> bool:
            return repo.repo_type.lower() == expected

        return _type_filter

    else:  # key == "refs"
        if op != "=":
            raise ValueError(f"Only '=' is supported for 'refs' filters. Got {op}.")

        def _refs_filter(repo: CachedRepoInfo, revision: CachedRevisionInfo | None, _: float) -> bool:
            refs = revision.refs if revision is not None else repo_refs_map.get(repo, frozenset())
            return value_raw.lower() in [ref.lower() for ref in refs]

        return _refs_filter


def _compare_numeric(left: float | None, op: str, right: float) -> bool:
    """Evaluate numeric comparisons for filters."""
    if left is None:
        return False

    comparisons = {
        "=": left == right,
        "!=": left != right,
        ">": left > right,
        "<": left < right,
        ">=": left >= right,
        "<=": left <= right,
    }

    if op not in comparisons:
        raise ValueError(f"Unsupported numeric comparison operator: {op}")

    return comparisons[op]


def compile_cache_sort(sort_expr: str) -> tuple[Callable[[CacheEntry], tuple[Any, ...]], bool]:
    """Convert a `mega cache ls` sort expression into a key function for sorting entries.

    Returns:
        A tuple of (key_function, reverse_flag) where reverse_flag indicates whether
        to sort in descending order (True) or ascending order (False).
    """
    match = _SORT_PATTERN.match(sort_expr.strip().lower())
    if not match:
        raise ValueError(f"Invalid sort expression: '{sort_expr}'. Expected format: 'key' or 'key:asc' or 'key:desc'.")

    key = match.group("key").lower()
    explicit_order = match.group("order")

    if key not in _SORT_KEYS:
        raise ValueError(f"Unsupported sort key '{key}' in '{sort_expr}'. Must be one of {list(_SORT_KEYS)}.")

    # Use explicit order if provided, otherwise use default for the key
    order = explicit_order if explicit_order else _SORT_DEFAULT_ORDER[key]
    reverse = order == "desc"

    def _sort_key(entry: CacheEntry) -> tuple[Any, ...]:
        repo, revision = entry

        if key == "name":
            # Sort by cache_id (repo type/id)
            value: Any = repo.cache_id.lower()
            return (value,)

        if key == "size":
            # Use revision size if available, otherwise repo size
            value = revision.size_on_disk if revision is not None else repo.size_on_disk
            return (value,)

        if key == "accessed":
            # For revisions, accessed is not available per-revision, use repo's last_accessed
            # For repos, use repo's last_accessed
            value = repo.last_accessed if repo.last_accessed is not None else 0.0
            return (value,)

        if key == "modified":
            # Use revision's last_modified if available, otherwise repo's last_modified
            if revision is not None:
                value = revision.last_modified if revision.last_modified is not None else 0.0
            else:
                value = repo.last_modified if repo.last_modified is not None else 0.0
            return (value,)

        # Should never reach here due to validation above
        raise ValueError(f"Unsupported sort key: {key}")

    return _sort_key, reverse


def _resolve_deletion_targets(cache_info: CacheInfo, targets: list[str]) -> _DeletionResolution:
    """Resolve the deletion targets into a deletion resolution."""
    repo_lookup, revision_lookup = build_cache_index(cache_info)

    selected: dict[CachedRepoInfo, set[CachedRevisionInfo]] = defaultdict(set)
    revisions: set[str] = set()
    missing: list[str] = []

    for raw_target in targets:
        target = raw_target.strip()
        if not target:
            continue
        lowered = target.lower()

        if re.fullmatch(r"[0-9a-fA-F]{40}", lowered):
            match = revision_lookup.get(lowered)
            if match is None:
                missing.append(raw_target)
                continue
            repo, revision = match
            selected[repo].add(revision)
            revisions.add(revision.commit_hash)
            continue

        matched_repo = repo_lookup.get(_repo_cache_id_from_target(target).lower())
        if matched_repo is None:
            missing.append(raw_target)
            continue

        for revision in matched_repo.revisions:
            selected[matched_repo].add(revision)
            revisions.add(revision.commit_hash)

    frozen_selected = {repo: frozenset(revs) for repo, revs in selected.items()}
    return _DeletionResolution(
        revisions=frozenset(revisions),
        selected=frozen_selected,
        missing=tuple(missing),
    )


#### Cache CLI commands


@cache_cli.command(
    "list | ls",
    examples=[
        "mega cache ls",
        "mega cache ls --revisions",
        'mega cache ls --filter "size>1GB" --limit 20',
        "mega cache ls --format json",
    ],
)
def ls(
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to scan (defaults to the MEGA cache).",
        ),
    ] = None,
    revisions: Annotated[
        bool,
        Option(
            help="Include revisions in the output instead of aggregated repositories.",
        ),
    ] = False,
    filter: Annotated[
        list[str] | None,
        Option(
            "-f",
            "--filter",
            help="Filter entries (e.g. 'size>1GB', 'type=model', 'accessed>7d'). Can be used multiple times.",
        ),
    ] = None,
    sort: Annotated[
        SortOptions | None,
        Option(
            help="Sort entries by key. Supported keys: 'accessed', 'modified', 'name', 'size'. "
            "Append ':asc' or ':desc' to explicitly set the order (e.g., 'modified:asc'). "
            "Defaults: 'accessed', 'modified', 'size' default to 'desc' (newest/biggest first); "
            "'name' defaults to 'asc' (alphabetical).",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Option(
            help="Limit the number of results returned. Returns only the top N entries after sorting.",
        ),
    ] = None,
) -> None:
    """List cached repositories or revisions."""
    try:
        cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    filters = filter or []

    entries, repo_refs_map = collect_cache_entries(cache_info, include_revisions=revisions)
    try:
        filter_fns = [compile_cache_filter(expr, repo_refs_map) for expr in filters]
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    now = time.time()
    for fn in filter_fns:
        entries = [entry for entry in entries if fn(entry[0], entry[1], now)]

    # Apply sorting if requested
    if sort:
        try:
            sort_key_fn, reverse = compile_cache_sort(sort.value)
            entries.sort(key=sort_key_fn, reverse=reverse)
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc

    # Apply limit if requested
    if limit is not None:
        if limit < 0:
            raise click.BadParameter(f"Limit must be a positive integer, got {limit}.")
        entries = entries[:limit]

    if revisions:
        items = [
            {
                "id": repo.cache_id,
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "revision": revision.commit_hash,
                "snapshot_path": str(revision.snapshot_path),
                "size": revision.size_on_disk_str,
                "last_modified": revision.last_modified_str,
                "refs": sorted(revision.refs),
            }
            for repo, revision in entries
            if revision is not None
        ]
        out.table(
            items,
            headers=["id", "revision", "size", "last_modified", "refs"],
            id_key="revision",
            alignments={"size": "right"},
        )
    else:
        items = [
            {
                "id": repo.cache_id,
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "size": repo.size_on_disk_str,
                "last_accessed": repo.last_accessed_str or "",
                "last_modified": repo.last_modified_str,
                "refs": sorted(repo_refs_map.get(repo, frozenset())),
            }
            for repo, _ in entries
        ]
        out.table(
            items,
            headers=["id", "size", "last_accessed", "last_modified", "refs"],
            id_key="id",
            alignments={"size": "right"},
        )

    if entries:
        unique_repos = {repo for repo, _ in entries}
        repo_count = len(unique_repos)
        if revisions:
            revision_count = sum(1 for _, rev in entries if rev is not None)
            total_size = sum(rev.size_on_disk for _, rev in entries if rev is not None)
        else:
            revision_count = sum(len(repo.revisions) for repo in unique_repos)
            total_size = sum(repo.size_on_disk for repo in unique_repos)
        out.text(
            ANSI.bold(
                f"\nFound {repo_count} repo(s) for a total of {revision_count} revision(s)"
                f" and {_format_size(total_size)} on disk."
            )
        )

    incomplete_files = cache_info.incomplete_files
    if incomplete_files:
        out.hint(
            f"Found {len(incomplete_files)} incomplete download(s) totalling "
            f"{_format_size(cache_info.incomplete_size_on_disk)}. "
            "Remove them with 'mega cache prune'."
        )


@cache_cli.command(
    examples=[
        "mega cache rm model/gpt2",
        "mega cache rm mega://models/openai-community/gpt2",
        "mega cache rm <revision_hash>",
        "mega cache rm model/gpt2 --dry-run",
        "mega cache rm model/gpt2 --yes",
    ],
)
def rm(
    targets: Annotated[
        list[str],
        Argument(
            help="One or more repo IDs (e.g. model/bert-base-uncased), repo-level mega:// URIs, or revision hashes to delete.",
        ),
    ],
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to scan (defaults to the MEGA cache).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Option(
            help="Preview deletions without removing anything.",
        ),
    ] = False,
) -> None:
    """Remove cached repositories or revisions."""
    try:
        cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    resolution = _resolve_deletion_targets(cache_info, targets)

    if resolution.missing:
        details = "\n".join(f"  - {entry}" for entry in resolution.missing)
        out.warning(f"Could not find in cache:\n{details}")

    if len(resolution.revisions) == 0:
        out.text("Nothing to delete.")
        raise click.exceptions.Exit(code=0)

    strategy = cache_info.delete_revisions(*sorted(resolution.revisions))
    counts = summarize_deletions(resolution.selected)

    summary_parts: list[str] = []
    if counts.repo_count:
        summary_parts.append(f"{counts.repo_count} repo(s)")
    if counts.partial_revision_count:
        summary_parts.append(f"{counts.partial_revision_count} revision(s)")
    if not summary_parts:
        summary_parts.append(f"{counts.total_revision_count} revision(s)")

    summary_text = " and ".join(summary_parts)
    out.text(f"About to delete {summary_text} totalling {strategy.expected_freed_size_str}.")
    print_cache_selected_revisions(resolution.selected)

    if dry_run:
        out.result(
            "Dry run: no files were deleted.",
            dry_run=True,
            repos=counts.repo_count,
            revisions=counts.total_revision_count,
            size=strategy.expected_freed_size_str,
        )
        return

    out.confirm("Proceed with deletion?", yes=yes)

    strategy.execute()
    counts = summarize_deletions(resolution.selected)
    out.result(
        f"Deleted {counts.repo_count} repo(s) and {counts.total_revision_count} revision(s);"
        f" freed {strategy.expected_freed_size_str}.",
        repos_deleted=counts.repo_count,
        revisions_deleted=counts.total_revision_count,
        freed=strategy.expected_freed_size_str,
    )


@cache_cli.command(examples=["mega cache prune", "mega cache prune --dry-run"])
def prune(
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to scan (defaults to the MEGA cache).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        Option(
            "-y",
            "--yes",
            help="Skip confirmation prompt.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Option(
            help="Preview deletions without removing anything.",
        ),
    ] = False,
) -> None:
    """Remove detached revisions and incomplete downloads from the cache."""
    try:
        cache_info = scan_cache_dir(cache_dir)
    except CacheNotFound as exc:
        raise CLIError(f"Cache directory not found: {exc.cache_dir}") from exc

    selected: dict[CachedRepoInfo, frozenset[CachedRevisionInfo]] = {}
    revisions: set[str] = set()
    for repo in cache_info.repos:
        detached = frozenset(revision for revision in repo.revisions if len(revision.refs) == 0)
        if not detached:
            continue
        selected[repo] = detached
        revisions.update(revision.commit_hash for revision in detached)

    incomplete_files = cache_info.incomplete_files

    if len(revisions) == 0 and not incomplete_files:
        out.text("No unreferenced revisions or incomplete downloads found. Nothing to prune.")
        return

    strategy = cache_info.delete_revisions(*sorted(revisions))
    counts = summarize_deletions(selected)
    total_freed = strategy.expected_freed_size + cache_info.incomplete_size_on_disk

    summary = _prune_summary(counts.total_revision_count, len(incomplete_files))
    out.text(f"About to delete {summary} ({_format_size(total_freed)} total).")
    print_cache_selected_revisions(selected)

    if dry_run:
        out.result(
            "Dry run: no files were deleted.",
            dry_run=True,
            revisions=counts.total_revision_count,
            incomplete=len(incomplete_files),  # might be overstated but it's fine
            size=_format_size(total_freed),
        )
        return

    out.confirm("Proceed?", yes=yes)

    strategy.execute()
    for incomplete_file in incomplete_files:
        try:
            incomplete_file.file_path.unlink()
        except FileNotFoundError:
            pass  # already removed (e.g. by a full-repo deletion above)
        except OSError as exc:
            out.warning(f"Could not delete incomplete file {incomplete_file.file_path}: {exc}")
    out.result(
        f"Deleted {summary}; freed {_format_size(total_freed)}.",
        revisions_deleted=counts.total_revision_count,
        incomplete_deleted=len(incomplete_files),
        freed=_format_size(total_freed),
    )


@cache_cli.command(
    examples=[
        "mega cache verify mega/gpt2",
        "mega cache verify mega/gpt2 --revision refs/pr/1",
        "mega cache verify mega/my-dataset --repo-type dataset",
    ],
)
def verify(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoTypeOpt.model,
    revision: RevisionOpt = None,
    cache_dir: Annotated[
        str | None,
        Option(
            help="Cache directory to use when verifying files from cache (defaults to the MEGA cache).",
        ),
    ] = None,
    local_dir: Annotated[
        str | None,
        Option(
            help="If set, verify files under this directory instead of the cache.",
        ),
    ] = None,
    fail_on_missing_files: Annotated[
        bool,
        Option(
            "--fail-on-missing-files",
            help="Fail if some files exist on the remote but are missing locally.",
        ),
    ] = False,
    fail_on_extra_files: Annotated[
        bool,
        Option(
            "--fail-on-extra-files",
            help="Fail if some files exist locally but are not present on the remote revision.",
        ),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Verify SHA-256 checksums for a repo revision from cache or a local directory."""

    if local_dir is not None and cache_dir is not None:
        out.error("Cannot pass both --local-dir and --cache-dir. Use one or the other.")
        raise click.exceptions.Exit(code=2)

    repo_type_value = repo_type.value if hasattr(repo_type, "value") else str(repo_type)
    verified_path, remote_revision = _resolve_verification_root(
        repo_id=repo_id,
        repo_type=repo_type_value,
        revision=revision,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )
    local_by_path = {
        path.relative_to(verified_path).as_posix(): path for path in verified_path.rglob("*") if path.is_file()
    }
    remote_by_path = {
        item.path: item
        for item in MegaHubClient(token=token).list_files(repo_id, revision=remote_revision)
    }
    missing_paths = sorted(set(remote_by_path) - set(local_by_path))
    extra_paths = sorted(set(local_by_path) - set(remote_by_path))
    mismatches: list[tuple[str, str, str]] = []
    for path in sorted(set(remote_by_path) & set(local_by_path)):
        actual = _sha256_file(local_by_path[path])
        expected = remote_by_path[path].sha256.lower()
        if actual != expected:
            mismatches.append((path, expected, actual))

    exit_code = 0

    if mismatches:
        details = "\n".join(
            f"  - {path}: expected {expected} (sha256), got {actual}"
            for path, expected, actual in mismatches
        )
        out.text(f"Checksum verification failed for the following file(s):\n{details}")
        exit_code = 1

    if missing_paths:
        if fail_on_missing_files:
            details = "\n".join(f"  - {path}" for path in missing_paths)
            out.text(f"Missing files (present remotely, absent locally):\n{details}")
            exit_code = 1
        else:
            out.warning(
                f"{len(missing_paths)} remote file(s) are missing locally. "
                "Use --fail-on-missing-files for details."
            )

    if extra_paths:
        if fail_on_extra_files:
            details = "\n".join(f"  - {path}" for path in extra_paths)
            out.text(f"Extra files (present locally, absent remotely):\n{details}")
            exit_code = 1
        else:
            out.warning(
                f"{len(extra_paths)} local file(s) do not exist on the remote repo. "
                "Use --fail-on-extra-files for details."
            )

    if exit_code != 0:
        out.error(
            f"Verification failed for '{repo_id}' ({repo_type_value}) in {verified_path}.\n  Revision: {remote_revision}"
        )
        raise click.exceptions.Exit(code=exit_code)

    out.result(
        f"Verified {len(set(remote_by_path) & set(local_by_path))} file(s) for {repo_type_value} '{repo_id}'. All checksums match.",
        repo_id=repo_id,
        repo_type=repo_type_value,
        checked=len(set(remote_by_path) & set(local_by_path)),
        path=str(verified_path),
    )


def _resolve_verification_root(
    *,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    cache_dir: str | None,
    local_dir: str | None,
) -> tuple[Path, str]:
    if local_dir is not None:
        root = Path(local_dir).expanduser().resolve()
        if not root.is_dir():
            raise CLIError(f"Local directory does not exist: {root}")
        return root, revision or constants.DEFAULT_REVISION

    cache_root = Path(cache_dir or constants.MEGA_HUB_CACHE).expanduser().resolve()
    storage = cache_root / repo_folder_name(repo_id=repo_id, repo_type=repo_type)
    if not storage.is_dir():
        raise CLIError(f"Repository is not present in the MEGA cache: {storage}")

    if revision is not None:
        ref = storage / "refs" / revision
        commit = ref.read_text(encoding="utf-8").strip() if ref.is_file() else revision
    else:
        main_ref = storage / "refs" / constants.DEFAULT_REVISION
        if main_ref.is_file():
            commit = main_ref.read_text(encoding="utf-8").strip()
        else:
            snapshots = sorted(path.name for path in (storage / "snapshots").glob("*") if path.is_dir())
            if len(snapshots) != 1:
                raise CLIError("Cached revision is ambiguous. Pass --revision explicitly.")
            commit = snapshots[0]
    root = storage / "snapshots" / commit
    if not root.is_dir():
        raise CLIError(f"Cached snapshot does not exist: {root}")
    return root, commit


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

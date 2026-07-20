# SPDX-License-Identifier: Apache-2.0
"""MEGA-native repository commands for the bundled Hub CLI.

The command group deliberately follows the Worker API rather than exposing
upstream-only repository operations.  Each command maps to an implemented
MEGA endpoint through :class:`megatensors.hub.MegaHubClient`.
"""

import fnmatch
from functools import wraps
from typing import Annotated, Any, Callable, TypeVar

from megatensors.hub import MegaHubClient, MegaHubError
from megatensors._hub.errors import CLIError

from ._cp import make_cp
from ._cli_utils import RepoIdArg, RepoType, RepoTypeOpt, RevisionOpt, TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out


repos_cli = typer_factory(help="Manage MEGA Hub repositories.")
repos_cli.command("cp")(make_cp("repos"))
branch_cli = typer_factory(help="Manage MEGA repository branches.")
tag_cli = typer_factory(help="Manage MEGA repository tags.")
repos_cli.add_group(branch_cli, name="branch")
repos_cli.add_group(tag_cli, name="tag")


RepoTypeFilterOpt = Annotated[
    RepoType | None,
    Option("--type", "--repo-type", help="Filter by repository type."),
]
F = TypeVar("F", bound=Callable[..., Any])


def _worker_errors_as_cli_errors(command: F) -> F:
    """Present Worker API failures through the bundled CLI error formatter."""

    @wraps(command)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except MegaHubError as error:
            raise CLIError(str(error)) from error

    return wrapped  # type: ignore[return-value]


@repos_cli.command(
    "list | ls",
    examples=[
        "mega repos list",
        "mega repos list --type dataset",
        "mega repos list --owner research --search vision",
    ],
)
@_worker_errors_as_cli_errors
def repo_list(
    limit: Annotated[int, Option("--limit", help="Maximum repositories to list.", min=1)] = 100,
    repo_type: RepoTypeFilterOpt = None,
    owner: Annotated[str | None, Option("--owner", help="Filter by owner or organization.")] = None,
    search: Annotated[str | None, Option("--search", help="Search repository id, description, or tags.")] = None,
    token: TokenOpt = None,
) -> None:
    """List MEGA Hub repositories visible to the caller."""
    repos = MegaHubClient(token=token).list_repos(
        limit=limit,
        repo_type=repo_type.value if repo_type else None,
        owner=owner,
        search=search,
    )
    out.table(
        [
            {
                "id": repo.repo_id,
                "type": repo.repo_type,
                "visibility": _visibility(repo),
                "updated_at": repo.updated_at,
            }
            for repo in repos
        ],
        id_key="id",
    )


@repos_cli.command(
    "create",
    examples=[
        "mega repos create mega/my-model",
        "mega repos create mega/my-dataset --type dataset --private --tag vision",
        "mega repos create org/my-space --type space --description 'Demo application'",
    ],
)
@_worker_errors_as_cli_errors
def repo_create(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOpt = RepoType.model,
    private: Annotated[bool, Option("--private", help="Create a private repository.")] = False,
    public: Annotated[bool, Option("--public", help="Create a public repository.")] = False,
    description: Annotated[str, Option("--description", help="Repository description.")] = "",
    tag: Annotated[list[str] | None, Option("--tag", help="Repository tag. Repeatable.")] = None,
    license: Annotated[str, Option("--license", help="Repository license identifier.")] = "",
    exist_ok: Annotated[bool, Option("--exist-ok", help="Do not fail if the repository already exists.")] = False,
    token: TokenOpt = None,
) -> None:
    """Create a repository on MEGA Hub."""
    if private and public:
        raise CLIError("Cannot pass both --private and --public.")
    repo = MegaHubClient(token=token).create_repo(
        repo_id,
        repo_type=repo_type.value,
        private=private and not public,
        description=description,
        tags=tag,
        license=license,
        exist_ok=exist_ok,
    )
    out.result(
        "Repository created",
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
        visibility=_visibility(repo),
    )


@repos_cli.command("info", examples=["mega repos info mega/my-model"])
@_worker_errors_as_cli_errors
def repo_info(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """Show repository metadata."""
    repo = MegaHubClient(token=token).repo_info(repo_id)
    out.dict(
        {
            "repo_id": repo.repo_id,
            "repo_type": repo.repo_type,
            "visibility": _visibility(repo),
            "owner": repo.owner or None,
            "description": repo.description or None,
            "tags": list(repo.tags),
            "license": repo.license or None,
            "downloads": repo.downloads,
            "likes": repo.likes,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at,
        },
        id_key="repo_id",
    )


@repos_cli.command(
    "settings",
    examples=[
        "mega repos settings mega/my-model --description 'Release notes' --tag vision",
        "mega repos settings mega/my-model --private",
        "mega repos settings mega/my-model --public",
    ],
)
@_worker_errors_as_cli_errors
def repo_settings(
    repo_id: RepoIdArg,
    private: Annotated[bool | None, Option("--private", help="Make the repository private.")] = None,
    public: Annotated[bool, Option("--public", help="Make the repository public.")] = False,
    description: Annotated[str | None, Option("--description", help="Replace the repository description.")] = None,
    tag: Annotated[list[str] | None, Option("--tag", help="Replace repository tags. Repeatable.")] = None,
    license: Annotated[str | None, Option("--license", help="Replace the repository license identifier.")] = None,
    token: TokenOpt = None,
) -> None:
    """Update repository visibility and metadata."""
    if private is True and public:
        raise CLIError("Cannot pass both --private and --public.")
    visibility = False if public else private
    if visibility is None and description is None and tag is None and license is None:
        raise CLIError("Provide at least one field to update.")
    repo = MegaHubClient(token=token).update_repo(
        repo_id,
        private=visibility,
        description=description,
        tags=tag,
        license=license,
    )
    out.result(
        "Repository updated",
        repo_id=repo.repo_id,
        visibility=_visibility(repo),
        description=repo.description or None,
        tags=list(repo.tags),
        license=repo.license or None,
    )


@repos_cli.command("move", examples=["mega repos move mega/old-name mega/new-name"])
@_worker_errors_as_cli_errors
def repo_move(
    from_id: Annotated[str, Argument(help="Current repository id in 'namespace/name' form.")],
    to_id: Annotated[str, Argument(help="Destination repository id in 'namespace/name' form.")],
    token: TokenOpt = None,
) -> None:
    """Rename a repository or transfer it to an administered namespace."""
    repo = MegaHubClient(token=token).move_repo(from_id, to_id)
    out.result("Repository moved", from_id=from_id, repo_id=repo.repo_id, owner=repo.owner)


@repos_cli.command("duplicate", examples=["mega repos duplicate mega/source mega/copy"])
@_worker_errors_as_cli_errors
def repo_duplicate(
    from_id: Annotated[str, Argument(help="Source repository id in 'namespace/name' form.")],
    to_id: Annotated[str, Argument(help="Destination repository id in 'namespace/name' form.")],
    private: Annotated[bool | None, Option("--private", help="Make the destination private.")] = None,
    public: Annotated[bool, Option("--public", help="Make the destination public.")] = False,
    token: TokenOpt = None,
) -> None:
    """Duplicate repository history and files without transferring object bytes."""
    if private is True and public:
        raise CLIError("Cannot pass both --private and --public.")
    visibility = False if public else private
    repo = MegaHubClient(token=token).duplicate_repo(from_id, to_id, private=visibility)
    out.result(
        "Repository duplicated",
        from_id=from_id,
        repo_id=repo.repo_id,
        visibility=_visibility(repo),
    )


@branch_cli.command("list | ls", examples=["mega repos branch list mega/my-model"])
@_worker_errors_as_cli_errors
def branch_list(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """List repository branches."""
    refs = MegaHubClient(token=token).list_refs(repo_id)
    out.table(
        [
            {"name": ref.name, "target_revision": ref.target_revision, "ref": ref.ref, "updated_at": ref.updated_at}
            for ref in refs.branches
        ],
        id_key="name",
    )


@branch_cli.command(
    "create",
    examples=["mega repos branch create mega/my-model dev", "mega repos branch create mega/my-model dev --revision main"],
)
@_worker_errors_as_cli_errors
def branch_create(
    repo_id: RepoIdArg,
    branch: Annotated[str, Argument(help="Branch name.")],
    revision: RevisionOpt = "main",
    exist_ok: Annotated[bool, Option("--exist-ok", help="Do not fail if the branch already exists.")] = False,
    token: TokenOpt = None,
) -> None:
    """Create a branch from an existing revision."""
    ref = MegaHubClient(token=token).create_branch(
        repo_id, branch, revision=revision, exist_ok=exist_ok
    )
    out.result("Branch created", repo_id=repo_id, branch=ref.name, revision=ref.target_revision)


@branch_cli.command("delete", examples=["mega repos branch delete mega/my-model dev --yes"])
@_worker_errors_as_cli_errors
def branch_delete(
    repo_id: RepoIdArg,
    branch: Annotated[str, Argument(help="Branch name.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a repository branch."""
    out.confirm(f"Delete branch '{branch}' from '{repo_id}'?", yes=yes)
    MegaHubClient(token=token).delete_branch(repo_id, branch)
    out.result("Branch deleted", repo_id=repo_id, branch=branch)


@tag_cli.command("list | ls", examples=["mega repos tag list mega/my-model"])
@_worker_errors_as_cli_errors
def tag_list(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """List repository tags."""
    refs = MegaHubClient(token=token).list_refs(repo_id)
    out.table(
        [
            {
                "name": ref.name,
                "target_revision": ref.target_revision,
                "ref": ref.ref,
                "message": ref.message,
                "updated_at": ref.updated_at,
            }
            for ref in refs.tags
        ],
        id_key="name",
    )


@tag_cli.command(
    "create",
    examples=["mega repos tag create mega/my-model v1.0", "mega repos tag create mega/my-model v1.0 --message 'First release'"],
)
@_worker_errors_as_cli_errors
def tag_create(
    repo_id: RepoIdArg,
    tag: Annotated[str, Argument(help="Tag name.")],
    revision: RevisionOpt = "main",
    message: Annotated[str | None, Option("--message", "-m", help="Tag message.")] = None,
    exist_ok: Annotated[bool, Option("--exist-ok", help="Do not fail if the tag already exists.")] = False,
    token: TokenOpt = None,
) -> None:
    """Create an immutable tag for a repository revision."""
    ref = MegaHubClient(token=token).create_tag(
        repo_id, tag, revision=revision, message=message, exist_ok=exist_ok
    )
    out.result("Tag created", repo_id=repo_id, tag=ref.name, revision=ref.target_revision)


@tag_cli.command("delete", examples=["mega repos tag delete mega/my-model v1.0 --yes"])
@_worker_errors_as_cli_errors
def tag_delete(
    repo_id: RepoIdArg,
    tag: Annotated[str, Argument(help="Tag name.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a repository tag."""
    out.confirm(f"Delete tag '{tag}' from '{repo_id}'?", yes=yes)
    MegaHubClient(token=token).delete_tag(repo_id, tag)
    out.result("Tag deleted", repo_id=repo_id, tag=tag)


@repos_cli.command("files", examples=["mega repos files mega/my-model --revision main"])
@_worker_errors_as_cli_errors
def repo_files(
    repo_id: RepoIdArg,
    revision: RevisionOpt = "main",
    token: TokenOpt = None,
) -> None:
    """List files in a repository revision."""
    files = MegaHubClient(token=token).list_files(repo_id, revision=revision)
    out.table(
        [
            {
                "path": item.path,
                "size": item.size,
                "sha256": item.sha256,
                "content_type": item.content_type,
            }
            for item in files
        ],
        id_key="path",
        alignments={"size": "right"},
    )


@repos_cli.command(
    "delete-files",
    examples=[
        "mega repos delete-files mega/my-model config.json --yes",
        "mega repos delete-files mega/my-model '*.json' --revision main --yes",
        "mega repos delete-files mega/my-model --path artifacts/ --yes",
    ],
)
@_worker_errors_as_cli_errors
def repo_delete_files(
    repo_id: RepoIdArg,
    patterns: Annotated[list[str] | None, Argument(help="File paths or glob patterns to delete.")] = None,
    path_in_repo: Annotated[list[str] | None, Option("--path", help="Path or glob to delete. Repeatable.")] = None,
    revision: RevisionOpt = "main",
    commit_message: Annotated[str | None, Option("--commit-message", help="Commit message for the deletion.")] = None,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete one or more files from a repository."""
    requested = [*(patterns or []), *(path_in_repo or [])]
    if not requested:
        raise CLIError("Provide at least one file path or glob pattern.")

    api = MegaHubClient(token=token)
    matched = sorted(
        {
            item.path
            for item in api.list_files(repo_id, revision=revision)
            for pattern in requested
            if _match_repo_path(item.path, pattern)
        }
    )
    if not matched:
        out.result("No files matched", repo_id=repo_id)
        return

    out.confirm(f"Permanently delete {len(matched)} file(s) from '{repo_id}'?", yes=yes)
    refs = api.list_refs(repo_id)
    parent_revision = next((ref.target_revision for ref in refs.branches if ref.name == revision), None)
    if parent_revision is None:
        raise CLIError(f"Branch '{revision}' not found on '{repo_id}'.")
    result = api.create_commit(
        repo_id,
        [{"operation": "delete", "path": path} for path in matched],
        revision=revision,
        parent_revision=parent_revision,
        commit_message=commit_message or f"Delete {len(matched)} files",
    )
    out.result("Files deleted", repo_id=repo_id, count=len(matched), revision=result.get("revision"))


@repos_cli.command(
    "history | log",
    examples=["mega repos history mega/my-model --limit 20", "mega repos history mega/my-model --revision release"],
)
@_worker_errors_as_cli_errors
def repo_history(
    repo_id: RepoIdArg,
    revision: RevisionOpt = "main",
    limit: Annotated[int, Option("--limit", help="Maximum commits to display.", min=1)] = 50,
    token: TokenOpt = None,
) -> None:
    """Show the ancestry of a repository revision."""
    commits, next_cursor = MegaHubClient(token=token).list_commits(
        repo_id, revision=revision, limit=limit
    )
    out.table(
        [
            {
                "revision": commit.revision,
                "parent_revision": commit.parent_revision,
                "author": commit.author,
                "author_email": commit.author_email,
                "signature_status": commit.signature_status,
                "signer_fingerprint": commit.signer_fingerprint,
                "signer_subject": commit.signer_subject,
                "created_at": commit.created_at,
                "message": commit.message,
            }
            for commit in commits
        ],
        id_key="revision",
    )
    if next_cursor:
        out.hint(f"More history is available after cursor {next_cursor}.")


@repos_cli.command(
    "commit | commit-info",
    examples=["mega repos commit mega/my-model main", "mega repos commit mega/my-model a1b2c3d4"],
)
@_worker_errors_as_cli_errors
def repo_commit(
    repo_id: RepoIdArg,
    revision: Annotated[str, Argument(help="Branch, tag, or commit revision.")] = "main",
    token: TokenOpt = None,
) -> None:
    """Show commit metadata and changed files."""
    commit = MegaHubClient(token=token).get_commit(repo_id, revision)
    out.dict(
        {
            "revision": commit.revision,
            "parent_revision": commit.parent_revision,
            "message": commit.message,
            "author": commit.author,
            "author_email": commit.author_email,
            "signature_status": commit.signature_status,
            "signer_fingerprint": commit.signer_fingerprint,
            "signer_subject": commit.signer_subject,
            "created_at": commit.created_at,
            "files": [
                {
                    "path": change.path,
                    "change": change.change,
                    "size": change.size,
                    "sha256": change.sha256,
                    "previous_sha256": change.previous_sha256,
                }
                for change in commit.files
            ],
        },
        id_key="revision",
    )


@repos_cli.command("delete", examples=["mega repos delete mega/my-model --yes"])
@_worker_errors_as_cli_errors
def repo_delete(
    repo_id: RepoIdArg,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
    missing_ok: Annotated[bool, Option("--missing-ok", help="Do not fail if the repository does not exist.")] = False,
    token: TokenOpt = None,
) -> None:
    """Permanently delete a repository and its metadata."""
    out.confirm(f"Permanently delete repository '{repo_id}'?", yes=yes)
    try:
        MegaHubClient(token=token).delete_repo(repo_id)
    except MegaHubError as error:
        if missing_ok and str(error).startswith("404 "):
            out.result("Repository already absent", repo_id=repo_id)
            return
        raise
    out.result("Repository deleted", repo_id=repo_id)


def _visibility(repo: Any) -> str:
    return "private" if repo.private else "public"


def _match_repo_path(path: str, pattern: str) -> bool:
    normalized = pattern.rstrip("/")
    if pattern.endswith("/"):
        return path == normalized or path.startswith(f"{normalized}/")
    return path == pattern or fnmatch.fnmatch(path, pattern)


repo_files_cli = typer_factory(help="Manage files in a MEGA Hub repository.")


@repo_files_cli.command(
    "delete",
    examples=[
        "mega repo-files delete mega/my-model config.json --yes",
        "mega repo-files delete mega/my-model '*.json' --revision main --yes",
    ],
)
def repo_files_delete(
    repo_id: RepoIdArg,
    patterns: Annotated[
        list[str],
        Argument(help="File paths or glob patterns to delete. '*' matches recursively."),
    ],
    revision: RevisionOpt = "main",
    commit_message: Annotated[
        str | None,
        Option("--commit-message", help="Commit message for the deletion."),
    ] = None,
    yes: Annotated[
        bool,
        Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Delete repository files through the MEGA Worker API."""
    repo_delete_files(
        repo_id,
        patterns=patterns,
        revision=revision,
        commit_message=commit_message,
        yes=yes,
        token=token,
    )

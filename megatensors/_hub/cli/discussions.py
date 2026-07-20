# SPDX-License-Identifier: Apache-2.0
"""Repository community commands for the MEGA Hub CLI.

The command vocabulary follows ``hf discussions`` where the MEGA service has a
matching contract. Transport and response handling stay in ``MegaHubClient`` so
the CLI never depends on the third-party reference snapshot at runtime.
"""

import enum
import sys
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Callable, TypeVar

import click

from megatensors._hub.errors import CLIError
from megatensors.hub import DiscussionInfo, DiscussionMessageInfo, DiscussionThread, MegaHubClient, MegaHubError

from ._cli_utils import RepoIdArg, RepoType, RepoTypeOpt, TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out


class DiscussionStatus(str, enum.Enum):
    open = "open"
    closed = "closed"
    merged = "merged"
    all = "all"


class DiscussionKind(str, enum.Enum):
    all = "all"
    discussion = "discussion"
    pull_request = "pull_request"


class DiscussionSort(str, enum.Enum):
    created = "recently-created"
    updated = "recently-updated"


DiscussionNumArg = Annotated[
    int,
    Argument(help="The discussion or pull request number.", min=1),
]
MessageIdArg = Annotated[
    str,
    Argument(help="Message ID shown by `mega discussions info ... --format json`."),
]
F = TypeVar("F", bound=Callable[..., Any])


def _community_errors_as_cli_errors(command: F) -> F:
    @wraps(command)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except (MegaHubError, ValueError) as error:
            raise CLIError(str(error)) from error

    return wrapped  # type: ignore[return-value]


def _read_body(body: str | None, body_file: Path | None, *, required: bool) -> str | None:
    """Resolve shared HF-style ``--body`` / ``--body-file`` input."""
    if body is not None and body_file is not None:
        raise click.BadParameter("Cannot use both --body and --body-file.")
    if body_file is not None:
        value = sys.stdin.read() if str(body_file) == "-" else body_file.read_text(encoding="utf-8")
    else:
        value = body
    if required and (value is None or not value.strip()):
        raise click.BadParameter("Either --body or --body-file is required.")
    return value


def _discussion_record(discussion: DiscussionInfo) -> dict[str, Any]:
    pull_request = discussion.pull_request
    return {
        "num": discussion.number,
        "title": discussion.title,
        "kind": discussion.kind,
        "is_pull_request": discussion.is_pull_request,
        "status": discussion.status,
        "author": discussion.author.handle,
        "replies": discussion.reply_count,
        "reactions": discussion.reaction_count,
        "created_at": discussion.created_at,
        "updated_at": discussion.updated_at,
        "url": discussion.url,
        "source_branch": pull_request.source_branch if pull_request else None,
        "target_branch": pull_request.target_branch if pull_request else None,
        "source_revision": pull_request.source_revision if pull_request else None,
        "target_revision": pull_request.target_revision if pull_request else None,
        "merged_at": pull_request.merged_at if pull_request else None,
        "merged_by": pull_request.merged_by if pull_request else None,
    }


def _message_record(message: DiscussionMessageInfo) -> dict[str, Any]:
    return {
        "id": message.message_id,
        "author": message.author.handle,
        "body": message.body,
        "is_original": message.is_original,
        "created_at": message.created_at,
        "updated_at": message.updated_at,
        "fire": message.reaction_count,
        "viewer_reacted": message.viewer_reacted,
        "can_edit": message.can_edit,
        "can_delete": message.can_delete,
    }


def _thread_record(thread: DiscussionThread) -> dict[str, Any]:
    return {
        "num": thread.discussion.number,
        "discussion": _discussion_record(thread.discussion),
        "messages": [_message_record(message) for message in thread.messages],
        "permissions": {
            "can_reply": thread.permissions.can_reply,
            "can_close": thread.permissions.can_close,
            "can_reopen": thread.permissions.can_reopen,
            "can_delete": thread.permissions.can_delete,
            "can_merge": thread.permissions.can_merge,
            "merge_blocked_reason": thread.permissions.merge_blocked_reason,
        },
    }


def _latest_message(thread: DiscussionThread) -> DiscussionMessageInfo:
    if not thread.messages:
        raise CLIError("The service returned a discussion without messages.")
    return thread.messages[-1]


discussions_cli = typer_factory(help="Manage repository discussions and pull requests on MEGA Hub.")


@discussions_cli.command(
    "list | ls",
    examples=[
        "mega discussions list mega/my-model",
        "mega discussions list mega/my-model --kind pull_request --status merged",
        "mega discussions list mega/my-dataset --type dataset --author alice --format json",
    ],
)
@_community_errors_as_cli_errors
def discussion_list(
    repo_id: RepoIdArg,
    status: Annotated[
        DiscussionStatus,
        Option("-s", "--status", help="Filter by status (open, closed, merged, all)."),
    ] = DiscussionStatus.open,
    kind: Annotated[
        DiscussionKind,
        Option("-k", "--kind", help="Filter by kind (discussion, pull_request, all)."),
    ] = DiscussionKind.all,
    author: Annotated[str | None, Option("--author", help="Filter by author handle.")] = None,
    limit: Annotated[int, Option("--limit", help="Maximum results to return.", min=1)] = 30,
    sort: Annotated[
        DiscussionSort,
        Option("--sort", help="Sort by recently-created or recently-updated."),
    ] = DiscussionSort.created,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """List discussions and pull requests on a repository."""
    api = MegaHubClient(token=token)
    api_status = "closed" if status == DiscussionStatus.merged else status.value
    page_number = 1
    selected: list[DiscussionInfo] = []
    while len(selected) < limit:
        page = api.list_discussions(
            repo_id,
            status=api_status,
            kind=kind.value,
            sort=sort.value,
            page=page_number,
            limit=min(50, max(1, limit)),
            repo_type=repo_type.value,
        )
        for discussion in page.discussions:
            if status == DiscussionStatus.merged and discussion.status != "merged":
                continue
            if author is not None and discussion.author.handle.casefold() != author.casefold():
                continue
            selected.append(discussion)
            if len(selected) >= limit:
                break
        if not page.has_more:
            break
        page_number += 1
    out.table(
        [_discussion_record(discussion) for discussion in selected],
        headers=["num", "title", "kind", "status", "author", "replies", "reactions", "updated_at"],
        id_key="num",
        alignments={"num": "right", "replies": "right", "reactions": "right"},
    )


@discussions_cli.command(
    "info",
    examples=[
        "mega discussions info mega/my-model 5",
        "mega discussions info mega/my-model 5 --format json",
    ],
)
@_community_errors_as_cli_errors
def discussion_info(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Show a discussion, its messages, reactions, and viewer permissions."""
    thread = MegaHubClient(token=token).get_discussion(repo_id, num, repo_type=repo_type.value)
    out.dict(_thread_record(thread), id_key="num")


@discussions_cli.command(
    "create",
    examples=[
        'mega discussions create mega/my-model --title "Bug report" --body "Steps to reproduce"',
        'mega discussions create mega/my-model --title "Detailed proposal" --body-file proposal.md',
        'mega discussions create mega/my-model --title "Ship feature" --pull-request --source-branch feature/x',
    ],
)
@_community_errors_as_cli_errors
def discussion_create(
    repo_id: RepoIdArg,
    title: Annotated[str, Option("--title", help="Discussion or pull request title.")],
    body: Annotated[str | None, Option("--body", help="Description in Markdown.")] = None,
    body_file: Annotated[
        Path | None,
        Option("--body-file", help="Read the description from a file. Use '-' for stdin."),
    ] = None,
    pull_request: Annotated[
        bool,
        Option("--pull-request", "--pr", help="Open a branch-backed pull request."),
    ] = False,
    source_branch: Annotated[
        str | None,
        Option("--source-branch", "--source", help="Source branch for a pull request."),
    ] = None,
    target_branch: Annotated[
        str,
        Option("--target-branch", "--target", help="Target branch for a pull request."),
    ] = "main",
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Open a discussion or pull request."""
    description = _read_body(body, body_file, required=False)
    if description is None or not description.strip():
        description = "Opened with the MEGA CLI."
    if pull_request and not source_branch:
        raise click.BadParameter("--source-branch is required with --pull-request.")
    if not pull_request and source_branch is not None:
        raise click.BadParameter("--source-branch requires --pull-request.")
    thread = MegaHubClient(token=token).create_discussion(
        repo_id,
        title=title,
        body=description,
        kind="pull_request" if pull_request else "discussion",
        source_branch=source_branch,
        target_branch=target_branch,
        repo_type=repo_type.value,
    )
    discussion = thread.discussion
    out.result(
        f"Created {'pull request' if pull_request else 'discussion'} #{discussion.number} on {repo_id}",
        num=discussion.number,
        url=discussion.url,
        ref=f"refs/pr/{discussion.number}" if pull_request else None,
    )


@discussions_cli.command(
    "comment",
    examples=['mega discussions comment mega/my-model 5 --body "Thanks for the report."'],
)
@_community_errors_as_cli_errors
def discussion_comment(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    body: Annotated[str | None, Option("--body", help="Comment text in Markdown.")] = None,
    body_file: Annotated[
        Path | None,
        Option("--body-file", help="Read the comment from a file. Use '-' for stdin."),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Comment on a discussion or pull request."""
    content = _read_body(body, body_file, required=True)
    assert content is not None
    thread = MegaHubClient(token=token).reply_to_discussion(
        repo_id,
        num,
        content,
        repo_type=repo_type.value,
    )
    message = _latest_message(thread)
    out.result(f"Commented on #{num} in {repo_id}", num=num, message_id=message.message_id)


@discussions_cli.command(
    "edit",
    examples=['mega discussions edit mega/my-model 5 <message-id> --body "Updated context."'],
)
@_community_errors_as_cli_errors
def discussion_edit(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    message_id: MessageIdArg,
    body: Annotated[str | None, Option("--body", help="Replacement Markdown content.")] = None,
    body_file: Annotated[
        Path | None,
        Option("--body-file", help="Read replacement content from a file. Use '-' for stdin."),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Edit a discussion message you own or moderate."""
    content = _read_body(body, body_file, required=True)
    assert content is not None
    MegaHubClient(token=token).edit_discussion_message(
        repo_id,
        num,
        message_id,
        content,
        repo_type=repo_type.value,
    )
    out.result(f"Edited message {message_id} on #{num}", num=num, message_id=message_id)


def _change_status(
    repo_id: str,
    num: int,
    status: str,
    *,
    comment: str | None,
    yes: bool,
    repo_type: RepoType,
    token: str | None,
) -> None:
    verb = "Close" if status == "closed" else "Reopen"
    out.confirm(f"{verb} #{num} on '{repo_id}'?", yes=yes)
    thread = MegaHubClient(token=token).update_discussion(
        repo_id,
        num,
        status=status,
        comment=comment,
        repo_type=repo_type.value,
    )
    past = "Closed" if status == "closed" else "Reopened"
    out.result(f"{past} #{num} in {repo_id}", num=num, status=thread.discussion.status)


@discussions_cli.command(
    "close",
    examples=["mega discussions close mega/my-model 5", 'mega discussions close mega/my-model 5 --comment "Resolved."'],
)
@_community_errors_as_cli_errors
def discussion_close(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[str | None, Option("--comment", help="Optional final comment.")] = None,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Close a discussion or pull request."""
    _change_status(repo_id, num, "closed", comment=comment, yes=yes, repo_type=repo_type, token=token)


@discussions_cli.command(
    "reopen",
    examples=["mega discussions reopen mega/my-model 5", 'mega discussions reopen mega/my-model 5 --comment "More context arrived."'],
)
@_community_errors_as_cli_errors
def discussion_reopen(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[str | None, Option("--comment", help="Optional reopening comment.")] = None,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Reopen a closed, non-merged discussion."""
    _change_status(repo_id, num, "open", comment=comment, yes=yes, repo_type=repo_type, token=token)


@discussions_cli.command(
    "rename",
    examples=['mega discussions rename mega/my-model 5 "Updated title"'],
)
@_community_errors_as_cli_errors
def discussion_rename(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    new_title: Annotated[str, Argument(help="The new title.")],
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Rename a discussion or pull request."""
    thread = MegaHubClient(token=token).update_discussion(
        repo_id,
        num,
        title=new_title,
        repo_type=repo_type.value,
    )
    out.result(f"Renamed #{num} in {repo_id}", num=num, title=thread.discussion.title)


@discussions_cli.command(
    "merge",
    examples=["mega discussions merge mega/my-model 5", 'mega discussions merge mega/my-model 5 --comment "Ready to ship."'],
)
@_community_errors_as_cli_errors
def discussion_merge(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    comment: Annotated[str | None, Option("--comment", help="Optional merge comment.")] = None,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Fast-forward a ready pull request."""
    out.confirm(f"Merge #{num} on '{repo_id}'?", yes=yes)
    thread = MegaHubClient(token=token).merge_pull_request(
        repo_id,
        num,
        comment=comment,
        repo_type=repo_type.value,
    )
    out.result(f"Merged #{num} in {repo_id}", num=num, status=thread.discussion.status)


@discussions_cli.command(
    "diff",
    examples=[
        "mega discussions diff mega/my-model 5",
        "mega discussions diff mega/my-model 5 --format json",
    ],
)
@_community_errors_as_cli_errors
def discussion_diff(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Show the commits and file changes carried by a pull request."""
    api = MegaHubClient(token=token)
    thread = api.get_discussion(repo_id, num, repo_type=repo_type.value)
    pull_request = thread.discussion.pull_request
    if pull_request is None:
        raise CLIError(f"Discussion #{num} is not a pull request.")
    current = pull_request.source_revision
    target = pull_request.target_revision
    visited: set[str] = set()
    commits: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    while current != target:
        if current in visited or len(visited) >= 1_000:
            raise CLIError("Pull request history is cyclic or exceeds 1,000 commits.")
        visited.add(current)
        commit = api.get_commit(repo_id, current)
        commits.append(
            {
                "revision": commit.revision,
                "parent_revision": commit.parent_revision,
                "message": commit.message,
                "author": commit.author,
                "created_at": commit.created_at,
            }
        )
        changes.extend(
            {
                "commit": commit.revision,
                "path": change.path,
                "change": change.change,
                "size": change.size,
                "sha256": change.sha256,
                "previous_sha256": change.previous_sha256,
            }
            for change in commit.files
        )
        if commit.parent_revision is None:
            raise CLIError("The pull request source no longer contains its recorded target revision.")
        current = commit.parent_revision
    out.dict(
        {
            "num": num,
            "repo_id": repo_id,
            "base_revision": target,
            "head_revision": pull_request.source_revision,
            "commit_count": len(commits),
            "commits": commits,
            "changes": changes,
        },
        id_key="head_revision",
    )


@discussions_cli.command("delete", examples=["mega discussions delete mega/my-model 5 --yes"])
@_community_errors_as_cli_errors
def discussion_delete(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Delete a non-merged discussion you own or moderate."""
    out.confirm(f"Delete #{num} on '{repo_id}'? This cannot be undone.", yes=yes)
    MegaHubClient(token=token).delete_discussion(repo_id, num, repo_type=repo_type.value)
    out.result(f"Deleted #{num} in {repo_id}", num=num, repo=repo_id)


@discussions_cli.command(
    "delete-comment",
    examples=["mega discussions delete-comment mega/my-model 5 <message-id> --yes"],
)
@_community_errors_as_cli_errors
def discussion_delete_comment(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    message_id: MessageIdArg,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Delete a non-original message you own or moderate."""
    out.confirm(f"Delete message {message_id} from #{num}?", yes=yes)
    MegaHubClient(token=token).delete_discussion_message(
        repo_id,
        num,
        message_id,
        repo_type=repo_type.value,
    )
    out.result(f"Deleted message {message_id} from #{num}", num=num, message_id=message_id)


@discussions_cli.command(
    "react",
    examples=[
        "mega discussions react mega/my-model 5 <message-id>",
        "mega discussions react mega/my-model 5 <message-id> --remove",
    ],
)
@_community_errors_as_cli_errors
def discussion_react(
    repo_id: RepoIdArg,
    num: DiscussionNumArg,
    message_id: MessageIdArg,
    remove: Annotated[bool, Option("--remove", help="Remove your fire reaction.")] = False,
    repo_type: RepoTypeOpt = RepoType.model,
    token: TokenOpt = None,
) -> None:
    """Add or remove the MEGA fire reaction on a message."""
    reaction = MegaHubClient(token=token).set_discussion_reaction(
        repo_id,
        num,
        message_id,
        active=not remove,
        repo_type=repo_type.value,
    )
    out.result(
        f"{'Removed' if remove else 'Added'} fire reaction on {message_id}",
        message_id=message_id,
        active=bool(reaction.get("viewer_reacted", not remove)),
        count=int(reaction.get("count", 0)),
    )

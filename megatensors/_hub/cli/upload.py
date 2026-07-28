# SPDX-License-Identifier: Apache-2.0
"""MEGA Hub upload command using the shared upload implementation."""

from pathlib import Path
from secrets import token_hex
import time
from typing import Annotated

import click

from megatensors._hub import CommitScheduler, MegaApi
from megatensors._hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects

from ._cli_utils import RepoIdArg, RepoType, RepoTypeOpt, TokenOpt
from ._framework import Argument, Option
from ._output import out


UPLOAD_EXAMPLES = [
    "mega upload mega/my-model ./model.safetensors",
    "mega upload mega/my-model ./weights weights --revision main",
    "mega upload mega/my-dataset ./data train --repo-type dataset --include '*.jsonl'",
    "mega upload mega/my-model ./checkpoint --commit-message 'epoch 34' --max-workers 4",
]

UPLOAD_LARGE_FOLDER_EXAMPLES = [
    "mega upload-large-folder mega/my-model ./large-model",
    "mega upload-large-folder mega/my-dataset ./data --repo-type dataset --revision v1.0",
]


def upload(
    repo_id: RepoIdArg,
    local_path: Annotated[
        Path,
        Argument(help="Local file or directory to upload."),
    ] = Path("."),
    path_in_repo: Annotated[
        str | None,
        Argument(help="Remote path. Defaults to the local basename."),
    ] = None,
    repo_type: RepoTypeOpt = RepoType.model,
    revision: Annotated[
        str,
        Option("--revision", "-r", help="Branch, tag, or commit revision."),
    ] = "main",
    private: Annotated[
        bool,
        Option("--private", help="Create the repository as private if it does not exist."),
    ] = False,
    commit_message: Annotated[
        str | None,
        Option("--commit-message", help="Commit message for uploaded files."),
    ] = None,
    commit_description: Annotated[
        str | None,
        Option("--commit-description", help="Markdown description stored with the commit."),
    ] = None,
    create_pr: Annotated[
        bool,
        Option("--create-pr", help="Upload on a new branch and open a native pull request against main."),
    ] = False,
    every: Annotated[
        float | None,
        Option("--every", help="Keep this local process running and commit changed files every N minutes."),
    ] = None,
    include: Annotated[
        list[str] | None,
        Option("--include", help="Glob to include when uploading a directory. Repeatable."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        Option("--exclude", help="Glob to exclude when uploading a directory. Repeatable."),
    ] = None,
    delete: Annotated[
        list[str] | None,
        Option("--delete", help="Remote glob to delete in the same folder commit. Repeatable."),
    ] = None,
    max_workers: Annotated[
        int | None,
        Option(
            "--max-workers",
            help="Compatibility option; native Xet manages upload concurrency.",
            min=1,
        ),
    ] = None,
    sync: Annotated[
        bool,
        Option("--sync", help="Mirror the local directory by deleting remote files missing locally."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Upload a file or directory to MEGA Hub."""
    if every is not None:
        schedule_upload(
            repo_id,
            local_path=local_path,
            path_in_repo=path_in_repo,
            repo_type=repo_type.value,
            revision=revision,
            private=private,
            commit_description=commit_description,
            include=include,
            exclude=exclude,
            delete=delete,
            create_pr=create_pr,
            every=every,
            token=token,
        )
        return
    run_upload(
        repo_id,
        local_path=local_path,
        path_in_repo=path_in_repo,
        repo_type=repo_type.value,
        revision=revision,
        private=private,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
        include=include,
        exclude=exclude,
        delete=delete,
        max_workers=max_workers,
        sync=sync,
        token=token,
    )


def schedule_upload(
    repo_id: str,
    *,
    local_path: Path,
    path_in_repo: str | None,
    repo_type: str,
    revision: str,
    private: bool,
    commit_description: str | None,
    include: list[str] | None,
    exclude: list[str] | None,
    delete: list[str] | None,
    create_pr: bool,
    every: float,
    token: str | None,
) -> None:
    """Run the HF-compatible local periodic uploader without inventing a cloud schedule."""
    if every <= 0:
        raise click.BadParameter("--every must be a positive number of minutes.", param_hint="every")
    if create_pr:
        raise click.BadParameter("--every cannot be combined with --create-pr.", param_hint="create_pr")
    if commit_description is not None:
        raise click.BadParameter(
            "--every cannot be combined with --commit-description because each scheduled commit needs its own description.",
            param_hint="commit_description",
        )
    if delete:
        raise click.BadParameter("--every cannot be combined with --delete.", param_hint="delete")

    if local_path.is_file():
        folder_path = local_path.parent
        remote_path = path_in_repo or local_path.name
        parent = Path(remote_path).parent.as_posix()
        scheduler_path = "" if parent == "." else parent
        allow_patterns = [local_path.name]
    else:
        folder_path = local_path
        scheduler_path = path_in_repo or ""
        allow_patterns = include

    api = MegaApi(token=token, library_name="mega-cli")
    scheduler = CommitScheduler(
        repo_id=repo_id,
        folder_path=folder_path,
        every=every,
        path_in_repo=scheduler_path,
        repo_type=repo_type,
        revision=revision,
        private=private,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=exclude,
        mega_api=api,
    )
    out.text(f"Scheduling local commits every {every:g} minutes to {scheduler.repo_id}. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        scheduler.stop()
        out.result("Stopped scheduled uploads", repo_id=scheduler.repo_id)


def run_upload(
    repo_id: str,
    *,
    local_path: Path = Path("."),
    path_in_repo: str | None = None,
    repo_type: str = "model",
    revision: str = "main",
    private: bool = False,
    commit_message: str | None = None,
    commit_description: str | None = None,
    create_pr: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    delete: list[str] | None = None,
    max_workers: int | None = None,
    sync: bool = False,
    token: str | None = None,
) -> None:
    """Upload through the HF-compatible native Xet and Git commit routes."""
    api = MegaApi(token=token, library_name="mega-cli")
    api.create_repo(
        repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    if create_pr and revision != "main":
        raise click.BadParameter("--create-pr can only target the default main branch.", param_hint="revision")
    target_revision = revision
    if create_pr:
        target_revision = f"upload-{token_hex(6)}"
        api.create_branch(repo_id, branch=target_revision, revision="main", repo_type=repo_type)

    if local_path.is_file():
        if delete:
            raise click.BadParameter("--delete requires a directory upload.", param_hint="delete")
        remote_path = (path_in_repo or local_path.name).strip("/")
        result = api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            revision=target_revision,
            commit_message=commit_message,
            commit_description=commit_description,
            repo_type=repo_type,
        )
        _report_upload(
            api,
            repo_id,
            repo_type,
            target_revision,
            result.oid,
            commit_message,
            commit_description,
            create_pr,
            path=remote_path,
        )
        return

    if not local_path.is_dir():
        raise click.BadParameter(
            "Local path must be a file or directory.", param_hint="local_path"
        )

    delete_patterns: list[str] | str | None = list(delete or []) or None
    if sync:
        delete_patterns = "**" if delete_patterns is None else ["**", *delete_patterns]
    result = api.upload_folder(
        repo_id=repo_id,
        folder_path=local_path,
        path_in_repo=path_in_repo or "",
        revision=target_revision,
        repo_type=repo_type,
        commit_message=commit_message,
        commit_description=commit_description,
        allow_patterns=list(include) if include else None,
        ignore_patterns=list(exclude) if exclude else None,
        delete_patterns=delete_patterns,
    )
    # Xet manages its own bounded upload concurrency; retain --max-workers as a
    # compatibility option while reporting the exact number of selected files.
    _ = max_workers
    relative_files = [
        path.relative_to(local_path).as_posix()
        for path in local_path.glob("**/*")
        if path.is_file()
    ]
    ignored = [*(exclude or []), *DEFAULT_IGNORE_PATTERNS]
    selected_files = list(
        filter_repo_objects(
            relative_files,
            allow_patterns=include,
            ignore_patterns=ignored,
        )
    )
    _report_upload(
        api,
        repo_id,
        repo_type,
        target_revision,
        result.oid,
        commit_message,
        commit_description,
        create_pr,
        files=len(selected_files),
    )


def _report_upload(
    api: MegaApi,
    repo_id: str,
    repo_type: str,
    branch: str,
    revision: str,
    commit_message: str | None,
    commit_description: str | None,
    create_pr: bool,
    *,
    path: str | None = None,
    files: int | None = None,
) -> None:
    if not create_pr:
        result = {"repo_id": repo_id, "revision": revision}
        if path is not None:
            result["path"] = path
        if files is not None:
            result["files"] = files
        out.result(
            "File uploaded" if path is not None else "Folder uploaded",
            **result,
        )
        return
    pull_request = api.create_pull_request(
        repo_id,
        commit_message or "Upload changes",
        description=commit_description or "Changes uploaded with the MEGA CLI.",
        repo_type=repo_type,
        source_branch=branch,
        target_branch="main",
    )
    out.result(
        "Pull request opened",
        repo_id=repo_id,
        branch=branch,
        revision=revision,
        pull_request=getattr(pull_request, "url", None),
    )


def upload_large_folder(
    repo_id: RepoIdArg,
    local_path: Annotated[Path, Argument(help="Local directory to upload.")],
    repo_type: RepoTypeOpt = RepoType.model,
    revision: Annotated[str, Option("--revision", "-r", help="Branch, tag, or commit revision.")] = "main",
    private: Annotated[
        bool,
        Option("--private", help="Create the repository as private if it does not exist."),
    ] = False,
    include: Annotated[
        list[str] | None,
        Option("--include", help="Glob to include. Repeatable."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        Option("--exclude", help="Glob to exclude. Repeatable."),
    ] = None,
    num_workers: Annotated[
        int | None,
        Option("--num-workers", help="Concurrency for directory uploads.", min=1),
    ] = None,
    sync: Annotated[
        bool,
        Option("--sync", help="Mirror the local directory by deleting remote files missing locally."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Upload a directory through MEGA's native Xet and atomic Git commit routes."""
    if not local_path.is_dir():
        raise click.BadParameter("Directory uploads require a local directory.", param_hint="local_path")
    run_upload(
        repo_id,
        local_path=local_path,
        repo_type=repo_type.value,
        revision=revision,
        private=private,
        include=include,
        exclude=exclude,
        max_workers=num_workers,
        sync=sync,
        token=token,
    )

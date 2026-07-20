# SPDX-License-Identifier: Apache-2.0
"""MEGA Storage Bucket commands aligned with ``huggingface_hub``."""

from typing import Annotated, Literal

import click

from megatensors._hub._buckets import BUCKET_PREFIX, BucketFile, FilterMatcher, _parse_bucket_uri
from megatensors._hub.mega_api import MegaApi

from ._cli_utils import TokenOpt, typer_factory
from ._bucket_listing import print_bucket_listing
from ._cp import make_cp
from ._framework import Argument, Option
from ._output import out


BucketRegion = Literal["us", "eu"]
buckets_cli = typer_factory(help="Manage mutable Xet-backed Storage Buckets.")


@buckets_cli.command(
    "create",
    examples=[
        "mega buckets create my-bucket",
        "mega buckets create owner/my-bucket --private",
        "mega buckets create owner/my-bucket --region eu --exist-ok",
    ],
)
def create(
    bucket_id: Annotated[str, Argument(help="Bucket name, owner/name, or mega://buckets/owner/name.")],
    private: Annotated[bool, Option("--private", help="Create a private Bucket.")] = False,
    region: Annotated[BucketRegion | None, Option("--region", help="MEGA storage region: us or eu; the server must advertise it as writable.")] = None,
    exist_ok: Annotated[bool, Option("--exist-ok", help="Succeed when the Bucket already exists.")] = False,
    token: TokenOpt = None,
) -> None:
    """Create a Storage Bucket."""
    if bucket_id.startswith(BUCKET_PREFIX):
        parsed = _parse_bucket_uri(bucket_id)
        if parsed.path_in_repo:
            raise click.BadParameter("Bucket creation does not accept a file prefix.")
        bucket_id = parsed.id
    bucket = MegaApi(token=token).create_bucket(
        bucket_id,
        private=True if private else None,
        region=region,
        exist_ok=exist_ok,
    )
    out.result("Bucket created", uri=bucket.uri.to_uri(), url=bucket.url)


@buckets_cli.command(
    "list | ls",
    examples=[
        "mega buckets list",
        "mega buckets list owner",
        "mega buckets list owner/my-bucket -R",
        "mega buckets list mega://buckets/owner/my-bucket/checkpoints --human-readable",
    ],
)
def list_cmd(
    argument: Annotated[str | None, Argument(help="Namespace to list, or owner/Bucket[/prefix] to list files.")] = None,
    human_readable: Annotated[bool, Option("--human-readable", "-h", help="Render byte sizes with units.")] = False,
    as_tree: Annotated[bool, Option("--tree", help="Render Bucket files as a tree.")] = False,
    recursive: Annotated[bool, Option("--recursive", "-R", help="List Bucket files recursively.")] = False,
    search: Annotated[str | None, Option("--search", help="Filter Bucket names.")] = None,
    token: TokenOpt = None,
) -> None:
    """List Buckets in a namespace or files in one Bucket."""
    if argument is not None and _is_bucket_id(argument):
        if search is not None:
            raise click.BadParameter("--search applies only when listing Buckets.")
        parsed = _parse_bucket_uri(argument)
        entries = list(MegaApi(token=token).list_bucket_tree(
            parsed.id,
            prefix=parsed.path_in_repo or None,
            recursive=recursive,
        ))
        print_bucket_listing(
            entries,
            human_readable=human_readable,
            as_tree=as_tree,
            recursive=recursive,
        )
        return
    if as_tree:
        raise click.BadParameter("--tree requires a Bucket ID.")
    if recursive:
        raise click.BadParameter("--recursive requires a Bucket ID.")
    namespace = argument
    if namespace and namespace.startswith(BUCKET_PREFIX):
        namespace = namespace[len(BUCKET_PREFIX) :].rstrip("/")
    buckets = MegaApi(token=token).list_buckets(namespace=namespace, search=search)
    out.table(
        [
            {
                "id": bucket.id,
                "private": bucket.private,
                "size": _format_size(bucket.size, human_readable),
                "total_files": bucket.total_files,
                "created_at": bucket.created_at,
            }
            for bucket in buckets
        ],
        id_key="id",
        alignments={"size": "right", "total_files": "right"},
    )


@buckets_cli.command("info", examples=["mega buckets info owner/my-bucket"])
def info(
    bucket_id: Annotated[str, Argument(help="Bucket ID in owner/name form or a mega://buckets URI.")],
    token: TokenOpt = None,
) -> None:
    """Show Bucket metadata."""
    parsed = _parse_bucket_uri(bucket_id)
    if parsed.path_in_repo:
        raise click.BadParameter("Bucket info does not accept a file prefix.")
    bucket = MegaApi(token=token).bucket_info(parsed.id)
    out.dict(bucket, id_key="id")


@buckets_cli.command("delete", examples=["mega buckets delete owner/my-bucket", "mega buckets delete owner/my-bucket --yes"])
def delete(
    bucket_id: Annotated[str, Argument(help="Bucket ID in owner/name form or a mega://buckets URI.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    missing_ok: Annotated[bool, Option("--missing-ok", help="Succeed when the Bucket does not exist.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a Bucket and every file in it."""
    parsed = _parse_bucket_uri(bucket_id)
    if parsed.path_in_repo:
        raise click.BadParameter("Use `mega buckets rm` to remove files or prefixes.")
    out.confirm(f"Delete Bucket '{parsed.id}' and all of its files?", yes=yes)
    MegaApi(token=token).delete_bucket(parsed.id, missing_ok=missing_ok)
    out.result("Bucket deleted", bucket_id=parsed.id)


@buckets_cli.command(
    "remove | rm",
    examples=[
        "mega buckets rm owner/my-bucket/file.json --yes",
        "mega buckets rm owner/my-bucket/logs --recursive --dry-run",
    ],
)
def remove(
    argument: Annotated[str, Argument(help="Bucket file or prefix as owner/name/path or a mega://buckets URI.")],
    recursive: Annotated[bool, Option("--recursive", "-R", help="Remove all matching files below the prefix.")] = False,
    yes: Annotated[bool, Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    dry_run: Annotated[bool, Option("--dry-run", help="List files without deleting them.")] = False,
    include: Annotated[list[str] | None, Option("--include", help="Include matching paths; repeatable.")] = None,
    exclude: Annotated[list[str] | None, Option("--exclude", help="Exclude matching paths; repeatable.")] = None,
    token: TokenOpt = None,
) -> None:
    """Remove one file or a recursively selected prefix."""
    parsed = _parse_bucket_uri(argument)
    if not parsed.path_in_repo and not recursive:
        raise click.BadParameter("Specify a file path, or pass --recursive to target the whole Bucket.")
    if (include or exclude) and not recursive:
        raise click.BadParameter("--include and --exclude require --recursive.")
    api = MegaApi(token=token)
    if recursive:
        matcher = FilterMatcher(include_patterns=include, exclude_patterns=exclude)
        paths = [
            entry.path
            for entry in api.list_bucket_tree(parsed.id, prefix=parsed.path_in_repo or None, recursive=True)
            if isinstance(entry, BucketFile) and matcher.matches(entry.path)
        ]
    else:
        paths = [parsed.path_in_repo]
    if not paths:
        out.text("No files to remove.")
        return
    if dry_run:
        for path in paths:
            out.text(f"delete: {BUCKET_PREFIX}{parsed.id}/{path}")
        out.text(f"(dry run) {len(paths)} file(s) would be removed.")
        return
    out.confirm(f"Remove {len(paths)} file(s) from '{parsed.id}'?", yes=yes)
    api.batch_bucket_files(parsed.id, delete=paths)
    out.result("Bucket files removed", bucket_id=parsed.id, files_deleted=len(paths))


@buckets_cli.command("move", examples=["mega buckets move owner/old-name owner/new-name"])
def move(
    from_id: Annotated[str, Argument(help="Current owner/name Bucket ID.")],
    to_id: Annotated[str, Argument(help="Destination owner/name Bucket ID.")],
    token: TokenOpt = None,
) -> None:
    """Rename a Bucket or transfer it to another namespace."""
    source = _parse_bucket_uri(from_id)
    destination = _parse_bucket_uri(to_id)
    if source.path_in_repo or destination.path_in_repo:
        raise click.BadParameter("Bucket move accepts Bucket IDs without file prefixes.")
    MegaApi(token=token).move_bucket(source.id, destination.id)
    out.result("Bucket moved", from_id=source.id, to_id=destination.id)


@buckets_cli.command(
    "sync",
    examples=[
        "mega buckets sync ./data mega://buckets/owner/my-bucket",
        "mega buckets sync mega://buckets/owner/my-bucket ./data",
        "mega buckets sync ./data mega://buckets/owner/my-bucket --dry-run",
    ],
)
def sync(
    source: Annotated[str | None, Argument(help="Local directory or mega://buckets source.")] = None,
    dest: Annotated[str | None, Argument(help="Local directory or mega://buckets destination.")] = None,
    delete: Annotated[bool, Option("--delete", help="Delete destination files absent from the source.")] = False,
    ignore_times: Annotated[bool, Option("--ignore-times", help="Compare files without modification times.")] = False,
    ignore_sizes: Annotated[bool, Option("--ignore-sizes", help="Compare files without sizes.")] = False,
    existing: Annotated[bool, Option("--existing", help="Only update files already at the destination.")] = False,
    ignore_existing: Annotated[bool, Option("--ignore-existing", help="Only create files absent from the destination.")] = False,
    include: Annotated[list[str] | None, Option("--include", help="Include matching paths; repeatable.")] = None,
    exclude: Annotated[list[str] | None, Option("--exclude", help="Exclude matching paths; repeatable.")] = None,
    filter_from: Annotated[str | None, Option("--filter-from", help="Read ordered include/exclude rules from a file.")] = None,
    plan: Annotated[str | None, Option("--plan", help="Write the synchronization plan as JSONL.")] = None,
    apply: Annotated[str | None, Option("--apply", help="Apply a saved JSONL synchronization plan.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", help="Print the plan without changing files.")] = False,
    verbose: Annotated[bool, Option("--verbose", "-v", help="Show synchronization decisions.")] = False,
    token: TokenOpt = None,
) -> None:
    """Synchronize files between a local directory and a Bucket."""
    MegaApi(token=token).sync_bucket(
        source=source,
        dest=dest,
        delete=delete,
        ignore_times=ignore_times,
        ignore_sizes=ignore_sizes,
        existing=existing,
        ignore_existing=ignore_existing,
        include=include,
        exclude=exclude,
        filter_from=filter_from,
        plan=plan,
        apply=apply,
        dry_run=dry_run,
        verbose=verbose,
        quiet=out.is_quiet(),
    )
    if plan and not out.is_quiet():
        out.hint(f"Run `mega buckets sync --apply {plan}` after reviewing the plan.")


buckets_cli.command("cp", examples=["mega buckets cp ./file.json mega://buckets/owner/my-bucket/file.json"])(make_cp("buckets"))


def _is_bucket_id(argument: str) -> bool:
    value = argument[len(BUCKET_PREFIX) :] if argument.startswith(BUCKET_PREFIX) else argument
    return "/" in value


def _format_size(value: int, human_readable: bool) -> int | str:
    if not human_readable:
        return value
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    return f"{amount:.0f} {unit}" if amount >= 100 or unit == "B" else f"{amount:.1f} {unit}"

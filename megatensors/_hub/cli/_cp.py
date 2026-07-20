# SPDX-License-Identifier: Apache-2.0
"""Copy files between local paths and MEGA Hub repositories.

The command accepts local file paths, ``mega://`` repository URIs, and ``-``
for standard input or output. Remote-to-remote copies, including directories,
are executed by the MEGA Worker so the file bytes stay in object storage.
"""

import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, unquote

import click

from megatensors._hub.errors import CLIError
from megatensors._hub.mega_api import MegaApi
from megatensors.hub import MegaHubClient, MegaHubError

from ._cli_utils import TokenOpt
from ._framework import Argument
from ._output import out


MEGA_PROTOCOL = "mega://"
_TYPE_PREFIXES = {"models": "model", "datasets": "dataset", "spaces": "space", "buckets": "bucket", "bucket": "bucket"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


CP_EXAMPLES = [
    "mega cp mega://models/mega/source@main/config.json ./config.json",
    "mega cp mega://datasets/mega/data@v1/train.jsonl -",
    "mega cp ./model.mega mega://models/mega/demo@main/weights/model.mega",
    "mega cp - mega://models/mega/demo@main/config.json",
    "mega cp mega://models/mega/source@v2/model.mega mega://models/mega/destination@main/model.mega",
    "mega cp ./metrics.json mega://buckets/mega/artifacts/runs/metrics.json",
    "mega cp mega://models/mega/source@main/model.mega mega://buckets/mega/artifacts/model.mega",
]

CpContext = Literal["repos", "buckets"]


@dataclass(frozen=True)
class MegaUri:
    """A MEGA repository location parsed from a ``mega://`` URI."""

    repo_id: str
    repo_type: Literal["model", "dataset", "space", "bucket"]
    revision: str | None
    path: str = ""
    trailing_slash: bool = False

    def to_uri(self) -> str:
        if self.repo_type == "bucket":
            result = f"{MEGA_PROTOCOL}buckets/{self.repo_id}"
        else:
            prefix = {"model": "models", "dataset": "datasets", "space": "spaces"}[self.repo_type]
            revision = quote(self.revision or "main", safe="")
            result = f"{MEGA_PROTOCOL}{prefix}/{self.repo_id}@{revision}"
        if self.path:
            result += f"/{self.path}"
        if self.trailing_slash:
            result += "/"
        return result

    def with_path(self, path: str) -> "MegaUri":
        return MegaUri(self.repo_id, self.repo_type, self.revision, path)


def parse_mega_uri(value: str) -> MegaUri:
    """Parse a MEGA repository URI that identifies a file or directory."""
    if not value.startswith(MEGA_PROTOCOL):
        raise CLIError("Remote paths must use the `mega://` scheme.")

    body = value[len(MEGA_PROTOCOL) :]
    trailing_slash = body.endswith("/")
    if trailing_slash:
        body = body[:-1]
    segments = body.split("/")
    if any(segment == "" for segment in segments):
        raise _invalid_uri(value)

    repo_type: str = "model"
    if segments[0] in _TYPE_PREFIXES:
        repo_type = _TYPE_PREFIXES[segments.pop(0)]

    if len(segments) < 2:
        raise _invalid_uri(value)
    namespace, name_and_revision, *path_segments = segments
    if not _NAME_RE.fullmatch(namespace):
        raise _invalid_uri(value)

    name, marker, encoded_revision = name_and_revision.partition("@")
    if not _NAME_RE.fullmatch(name):
        raise _invalid_uri(value)
    if repo_type == "bucket" and marker:
        raise CLIError("Bucket URIs do not have revisions; remove the `@revision` suffix.")
    revision = None if repo_type == "bucket" else unquote(encoded_revision) if marker else "main"
    if revision is not None and (not revision or any(part in {"", ".", ".."} for part in revision.split("/"))):
        raise _invalid_uri(value)

    if any(part in {"", ".", ".."} for part in path_segments):
        raise _invalid_uri(value)
    path = "/".join(path_segments)

    return MegaUri(
        repo_id=f"{namespace}/{name}",
        repo_type=repo_type,  # type: ignore[arg-type]
        revision=revision,
        path=path,
        trailing_slash=trailing_slash,
    )


def make_cp(context: CpContext | None = None):
    """Build the shared copy command, optionally constrained to one resource family."""

    def cp(
        src: Annotated[str, Argument(help="Source: local file, `mega://` repository path, or `-` for standard input.")],
        dst: Annotated[
            str | None,
            Argument(help="Destination: local path, `mega://` repository path, or `-` for standard output."),
        ] = None,
        token: TokenOpt = None,
    ) -> None:
        """Copy files between local storage and MEGA Hub repositories."""
        _enforce_context(context, src, dst)
        _run_cp(src, dst, token=token)

    return cp


def _enforce_context(context: CpContext | None, src: str, dst: str | None) -> None:
    if context is None:
        return
    remote = dst if dst is not None and dst.startswith(MEGA_PROTOCOL) else src
    if not remote.startswith(MEGA_PROTOCOL):
        return
    is_bucket = parse_mega_uri(remote).repo_type == "bucket"
    if context == "repos" and is_bucket:
        raise CLIError("`mega repos cp` only works with repositories. Use `mega cp` or `mega buckets cp` for Buckets.")
    if context == "buckets" and not is_bucket:
        raise CLIError("`mega buckets cp` only works with Buckets. Use `mega cp` or `mega repos cp` for repositories.")


def _run_cp(src: str, dst: str | None, *, token: str | None) -> None:
    src_is_stdin = src == "-"
    dst_is_stdout = dst == "-"
    src_is_remote = src.startswith(MEGA_PROTOCOL)
    dst_is_remote = dst is not None and dst.startswith(MEGA_PROTOCOL)

    if "://" in src and not src_is_remote:
        raise CLIError("Remote paths must use the `mega://` scheme.")
    if dst is not None and "://" in dst and not dst_is_remote:
        raise CLIError("Remote paths must use the `mega://` scheme.")

    if not src_is_remote and not dst_is_remote:
        if dst is None:
            raise click.BadParameter("Missing destination. Provide a `mega://` repository file path as DST.")
        raise click.BadParameter("One of SRC or DST must be a `mega://` repository file path.")
    if src_is_stdin and dst_is_stdout:
        raise click.BadParameter("Standard input cannot be copied directly to standard output.")
    if dst is None and not src_is_remote:
        raise click.BadParameter("Missing destination. Provide a `mega://` repository file path as DST.")

    try:
        api = MegaHubClient(token=token)
        bucket_api = MegaApi(token=token)
        if src_is_remote and dst_is_remote:
            source = parse_mega_uri(src)
            destination = parse_mega_uri(dst)
            if source.repo_type == "bucket" or destination.repo_type == "bucket":
                if destination.repo_type != "bucket":
                    raise CLIError("Bucket-to-repository copy is not supported; publish an explicit repository commit instead.")
                bucket_api.copy_files(source.to_uri(), destination.to_uri())
                out.result("Copied", src=source.to_uri(), dst=destination.to_uri())
                return
            api.copy_files(
                source.repo_id,
                source.path,
                destination.repo_id,
                destination.path,
                source_revision=source.revision or "main",
                revision=destination.revision or "main",
                source_merge_contents=source.trailing_slash,
                destination_is_directory=destination.trailing_slash or destination.path == "",
            )
            out.result("Copied", src=source.to_uri(), dst=destination.to_uri())
            return

        if src_is_remote:
            source = parse_mega_uri(src)
            _require_file_source(source)
            if dst_is_stdout:
                _download_to_stdout(api, bucket_api, source)
            else:
                _download_to_local(api, bucket_api, source, dst)
            return

        assert dst is not None
        destination = parse_mega_uri(dst)
        _upload_to_remote(api, bucket_api, src, destination, src_is_stdin=src_is_stdin)
    except MegaHubError as error:
        raise CLIError(str(error)) from error


def _download_to_stdout(api: MegaHubClient, bucket_api: MegaApi, source: MegaUri) -> None:
    with tempfile.TemporaryDirectory(prefix="mega-cp-") as tmp_dir:
        if source.repo_type == "bucket":
            downloaded = Path(tmp_dir) / source.path.rsplit("/", 1)[-1]
            bucket_api.download_bucket_files(source.repo_id, [(source.path, downloaded)], raise_on_missing_files=True)
        else:
            downloaded = api.download_file(
                source.repo_id,
                source.path,
                local_dir=tmp_dir,
                revision=source.revision or "main",
                force=True,
            )
        with downloaded.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                sys.stdout.buffer.write(chunk)


def _download_to_local(api: MegaHubClient, bucket_api: MegaApi, source: MegaUri, destination: str | None) -> None:
    filename = source.path.rsplit("/", 1)[-1]
    if destination is None:
        local_path = Path(filename)
    elif os.path.isdir(destination) or destination.endswith((os.sep, "/")):
        local_path = Path(destination) / filename
    else:
        local_path = Path(destination)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".mega-cp-", dir=local_path.parent) as tmp_dir:
        if source.repo_type == "bucket":
            downloaded = Path(tmp_dir) / filename
            bucket_api.download_bucket_files(source.repo_id, [(source.path, downloaded)], raise_on_missing_files=True)
        else:
            downloaded = api.download_file(
                source.repo_id,
                source.path,
                local_dir=tmp_dir,
                revision=source.revision or "main",
                force=True,
            )
        os.replace(downloaded, local_path)
    out.result("Downloaded", src=source.to_uri(), dst=str(local_path))


def _upload_to_remote(
    api: MegaHubClient,
    bucket_api: MegaApi,
    source: str,
    destination: MegaUri,
    *,
    src_is_stdin: bool,
) -> None:
    if src_is_stdin:
        if not destination.path or destination.trailing_slash:
            raise click.BadParameter("Standard input uploads require a destination file path.")
        with tempfile.TemporaryDirectory(prefix="mega-cp-") as tmp_dir:
            local_path = Path(tmp_dir) / destination.path.rsplit("/", 1)[-1]
            with local_path.open("wb") as file:
                while chunk := sys.stdin.buffer.read(1024 * 1024):
                    file.write(chunk)
            if destination.repo_type == "bucket":
                bucket_api.batch_bucket_files(destination.repo_id, add=[(local_path.read_bytes(), destination.path)])
            else:
                api.upload_file(
                    destination.repo_id,
                    local_path,
                    path_in_repo=destination.path,
                    revision=destination.revision or "main",
                    repo_type=destination.repo_type,
                )
    else:
        local_path = Path(source)
        if local_path.is_dir():
            raise click.BadParameter("Source must be a file. Use `mega upload` for directories.")
        if not local_path.is_file():
            raise click.BadParameter(f"Source file not found: {source}")
    remote_path = destination.path
    if not remote_path or destination.trailing_slash:
        remote_path = "/".join(part for part in (remote_path.rstrip("/"), local_path.name) if part)
    if not src_is_stdin:
        if destination.repo_type == "bucket":
            bucket_api.batch_bucket_files(destination.repo_id, add=[(local_path, remote_path)])
        else:
            api.upload_file(
                destination.repo_id,
                local_path,
                path_in_repo=remote_path,
                revision=destination.revision or "main",
                repo_type=destination.repo_type,
            )
    out.result("Uploaded", src="stdin" if src_is_stdin else source, dst=destination.with_path(remote_path).to_uri())


def _require_file_source(source: MegaUri) -> None:
    if not source.path or source.trailing_slash:
        raise click.BadParameter("Remote downloads require a source file path. Use `mega download` for directories.")


def _invalid_uri(value: str) -> CLIError:
    return CLIError(
        "Invalid MEGA Hub URI. Expected a repository URI "
        "`mega://[models|datasets|spaces]/namespace/repository[@revision][/path]` or Bucket URI "
        "`mega://buckets/namespace/bucket[/path]`: "
        f"{value!r}."
    )

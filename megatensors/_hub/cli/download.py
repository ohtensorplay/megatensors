# Copyright 202-present, the MEGA Hub contributors.
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
"""Download MEGA repository files and cache-aware snapshots."""

import sys
import warnings
from pathlib import Path
from typing import Annotated

from megatensors._hub import constants
from megatensors._hub._snapshot_download import snapshot_download
from megatensors._hub.errors import CLIError, MegaUriError
from megatensors._hub.file_download import (
    DryRunFileInfo,
    mega_hub_download as _download_file,
)
from megatensors._hub.utils import _format_size, parse_mega_uri as _parse_repo_uri

from ._cli_utils import RepoIdArg, RepoType, RepoTypeOptionalOpt, RevisionOpt
from ._framework import Argument, Option
from ._output import OutputFormat, out

DOWNLOAD_EXAMPLES = [
    "mega download mega/my-model",
    "mega download mega/my-model config.json tokenizer.json",
    'mega download mega/my-model --include "*.safetensors" --exclude "*.bin"',
    "mega download mega/my-model --local-dir ./models/my-model",
    "mega download mega/my-dataset data/ --repo-type dataset",
    "mega download mega://datasets/mega/my-dataset@main/data/",
]

_MEGA_PROTOCOL = "mega://"
_MEGA_REPO_TYPES = {RepoType.model.value, RepoType.dataset.value, RepoType.space.value}


def _human_status(message: str) -> None:
    if out.mode == OutputFormat.human:
        print(message, file=sys.stderr, flush=True)


def _repo_type_label(repo_type: str) -> str:
    return {
        RepoType.model.value: "Model",
        RepoType.dataset.value: "Dataset",
        RepoType.space.value: "Space",
    }.get(repo_type, repo_type.title())


def _target_directory(*, local_dir: str | None, cache_dir: str | None) -> str:
    target = local_dir or cache_dir or constants.MEGA_HUB_CACHE
    return str(Path(target).expanduser().resolve())


def _parse_mega_uri(value: str):
    """Parse a MEGA URI with the shared repository path/revision grammar."""
    try:
        uri = _parse_repo_uri(value)
    except MegaUriError as error:
        raise CLIError(
            "Invalid MEGA repository URI. Expected "
            "`mega://[models|datasets|spaces]/namespace/repository[@revision][/path]`."
        ) from error
    if uri.is_bucket or uri.type not in _MEGA_REPO_TYPES:
        raise CLIError(
            "MEGA download supports model, dataset, and space repositories only."
        )
    return uri


def download(
    repo_id: RepoIdArg,
    filenames: Annotated[
        list[str] | None,
        Argument(
            help="Files to download (e.g. `config.json`, `data/metadata.jsonl`).",
        ),
    ] = None,
    repo_type: RepoTypeOptionalOpt = None,
    revision: RevisionOpt = None,
    include: Annotated[
        list[str] | None,
        Option(
            help="Glob patterns to include from files to download. eg: *.json",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        Option(
            help="Glob patterns to exclude from files to download.",
        ),
    ] = None,
    cache_dir: Annotated[
        str | None,
        Option(
            help="Directory where to save files.",
        ),
    ] = None,
    local_dir: Annotated[
        str | None,
        Option(
            help="If set, downloaded files are placed under this directory.",
        ),
    ] = None,
    force_download: Annotated[
        bool,
        Option(
            help="If True, the files will be downloaded even if they are already cached.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        Option(
            help="If True, perform a dry run without actually downloading the file.",
        ),
    ] = False,
    token: Annotated[
        str | None,
        Option(
            help="MEGA access token. Defaults to the token from `mega auth login` or `MEGA_TOKEN`."
        ),
    ] = None,
    max_workers: Annotated[
        int,
        Option(
            help="Maximum number of workers to use for downloading files. Default is 8.",
        ),
    ] = 8,
) -> None:
    """Download files from MEGA Hub."""
    if local_dir is not None and cache_dir is not None:
        raise CLIError(
            "Cannot use both `--local-dir` and `--cache-dir` at the same time. "
            "Use `--cache-dir` (or set the MEGA_HOME environment variable) for shared caching, "
            "or `--local-dir` for a one-off download to a specific directory."
        )

    # `repo_id` may be a plain repo id or a `mega://` URI (e.g.
    # `mega://datasets/my-org/my-dataset@v1.0/data/`).
    # When a URI is provided, it is authoritative for the repo type, revision and (optionally) file path,
    # so explicit `--repo-type` / `--revision` options are forbidden alongside it.
    # Repository arguments accept only the MEGA URI scheme.
    if "://" in repo_id and not repo_id.startswith(_MEGA_PROTOCOL):
        raise CLIError("Repository URIs must use the `mega://` scheme.")
    if repo_id.startswith(_MEGA_PROTOCOL):
        if repo_type is not None:
            raise CLIError(
                f"'--repo-type' cannot be used with a `mega://` URI ('{repo_id}')."
            )
        if revision is not None:
            raise CLIError(
                f"'--revision' cannot be used with a `mega://` URI ('{repo_id}')."
            )
        uri = _parse_mega_uri(repo_id)
        # The URI parser strips trailing slashes, but `mega download` uses a trailing '/' to denote a subfolder
        # download (e.g. `data/` -> `data/**`). Re-append it when the URI explicitly ended with '/' so a folder
        # URI keeps routing through the subfolder code path below.
        path_in_repo = uri.path_in_repo
        if path_in_repo and repo_id.endswith("/"):
            path_in_repo += "/"
        repo_id, repo_type_str, revision = uri.id, uri.type, uri.revision
        if path_in_repo:
            if filenames:
                raise CLIError(
                    f"Cannot combine a file path in the mega:// URI ('{path_in_repo}') with positional filenames {filenames}."
                )
            filenames = [path_in_repo]
    else:
        repo_type_str = (repo_type or RepoType.model).value

    filenames_list = filenames if filenames is not None else []
    subfolders = [f for f in filenames_list if f.endswith("/")]
    subfolder_patterns = [f"{f.rstrip('/')}/**" for f in subfolders]
    regular_filenames = [f for f in filenames_list if not f.endswith("/")]

    # Error if subfolder patterns are combined with --include/--exclude
    # Guide user to use --include instead of subfolder argument
    if len(subfolder_patterns) > 0:
        if include is not None and len(include) > 0:
            raise CLIError(
                f"Cannot combine subfolder argument ('{subfolders[0]}') with `--include`. "
                f'Please use `--include "{subfolders[0]}*"` instead.'
            )
        if exclude is not None and len(exclude) > 0:
            raise CLIError(
                f"Cannot combine subfolder argument ('{subfolders[0]}') with `--exclude`. "
                f'Please use `--include "{subfolders[0]}*"` with `--exclude` instead.'
            )

    # Warn user if patterns are ignored (only if regular filenames are provided)
    if len(regular_filenames) > 0:
        if include is not None and len(include) > 0:
            warnings.warn(
                "Ignoring `--include` since filenames have been explicitly set."
            )
        if exclude is not None and len(exclude) > 0:
            warnings.warn(
                "Ignoring `--exclude` since filenames have been explicitly set."
            )

    is_single_file = len(regular_filenames) == 1 and len(subfolder_patterns) == 0
    action = "Planning" if dry_run else "Downloading"
    artifact_label = _repo_type_label(repo_type_str)
    source = constants.ENDPOINT
    target_directory = _target_directory(local_dir=local_dir, cache_dir=cache_dir)
    if is_single_file:
        _human_status(
            f"{action} {artifact_label} file from {source} to directory: {target_directory}"
        )
    else:
        _human_status(
            f"{action} {artifact_label} from {source} to directory: {target_directory}"
        )

    def run_download() -> str | DryRunFileInfo | list[DryRunFileInfo]:
        # Single file to download (not a subfolder): use the cache-aware file downloader.
        if is_single_file:
            return _download_file(
                repo_id=repo_id,
                repo_type=repo_type_str,
                revision=revision,
                filename=regular_filenames[0],
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_dir=local_dir,
                library_name="mega-cli",
                dry_run=dry_run,
            )

        # Otherwise: use `snapshot_download` to ensure all files come from the same revision.
        if len(regular_filenames) == 0 and len(subfolder_patterns) == 0:
            # No filenames provided: use include/exclude patterns
            allow_patterns = include
            ignore_patterns = exclude
        else:
            # Combine regular filenames and subfolder patterns as allow_patterns
            allow_patterns = regular_filenames + subfolder_patterns
            ignore_patterns = None

        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type_str,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            force_download=force_download,
            cache_dir=cache_dir,
            token=token,
            local_dir=local_dir,
            library_name="mega-cli",
            max_workers=max_workers,
            dry_run=dry_run,
            per_file_progress=out.mode == OutputFormat.human,
        )

    def _print_result(result: str | DryRunFileInfo | list[DryRunFileInfo]) -> None:
        if isinstance(result, str):
            out.result(
                "File downloaded" if is_single_file else "Snapshot ready", path=result
            )
            return

        # Print dry run info
        if isinstance(result, DryRunFileInfo):
            result = [result]
        will_download = [r for r in result if r.will_download]
        out.text(
            f"[dry-run] Will download {len(will_download)} files"
            f" (out of {len(result)})"
            f" totalling {_format_size(sum(r.file_size for r in will_download))}."
        )
        items = [
            {
                "file": info.filename,
                "size": _format_size(info.file_size) if info.will_download else "-",
            }
            for info in sorted(result, key=lambda x: x.filename)
        ]
        out.table(items)

    _print_result(run_download())


def _format_pattern_note(include: list[str] | None, exclude: list[str] | None) -> str:
    parts = []
    if include:
        parts.append(f"include: {', '.join(include)}")
    if exclude:
        parts.append(f"exclude: {', '.join(exclude)}")
    return f" ({'; '.join(parts)})" if parts else ""


def snapshot(
    repo_id: RepoIdArg,
    repo_type: RepoTypeOptionalOpt = None,
    revision: RevisionOpt = None,
    include: Annotated[
        list[str] | None, Option(help="Glob patterns to include. Repeatable.")
    ] = None,
    exclude: Annotated[
        list[str] | None, Option(help="Glob patterns to exclude. Repeatable.")
    ] = None,
    cache_dir: Annotated[
        str | None, Option(help="Shared MEGA cache directory.")
    ] = None,
    local_dir: Annotated[
        str | None, Option(help="Materialize the snapshot in this directory.")
    ] = None,
    force_download: Annotated[
        bool, Option(help="Download files again even when cached.")
    ] = False,
    dry_run: Annotated[
        bool, Option(help="Show the files without downloading them.")
    ] = False,
    token: Annotated[
        str | None,
        Option(
            help="MEGA access token. Defaults to `MEGA_TOKEN` or the active MEGA login."
        ),
    ] = None,
    max_workers: Annotated[
        int, Option(help="Maximum number of concurrent downloads.", min=1)
    ] = 8,
) -> None:
    """Download a complete MEGA repository snapshot."""
    download(
        repo_id=repo_id,
        filenames=None,
        repo_type=repo_type,
        revision=revision,
        include=include,
        exclude=exclude,
        cache_dir=cache_dir,
        local_dir=local_dir,
        force_download=force_download,
        dry_run=dry_run,
        token=token,
        max_workers=max_workers,
    )


def run_download(
    repo_id: str,
    *,
    filenames: list[str] | None = None,
    repo_type: str = "model",
    local_dir: str | Path = ".",
    cache_dir: str | Path | None = None,
    revision: str = "main",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    max_workers: int = 8,
    token: str | None = None,
) -> None:
    """Run the cache-aware MEGA download workflow for typed repo commands."""
    download(
        repo_id=repo_id,
        filenames=filenames,
        repo_type=RepoType(repo_type),
        revision=revision,
        include=include,
        exclude=exclude,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_dir=str(local_dir),
        force_download=force,
        dry_run=dry_run,
        token=token,
        max_workers=max_workers,
    )

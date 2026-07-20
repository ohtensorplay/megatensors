# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed repository and runtime commands for MEGA models, datasets, and Spaces."""

import itertools
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Annotated, Any

import click

from megatensors._hub.errors import CLIError
from megatensors._hub.mega_api import MegaApi
from megatensors._hub.file_download import mega_hub_download
from megatensors._hub.repocard import DatasetCard, ModelCard, SpaceCard
from megatensors._hub._dataset_viewer import execute_raw_sql_query
from megatensors._hub._space_api import Volume
from megatensors._hub.utils import parse_mega_mount
from megatensors._hub.utils._dotenv import load_dotenv
from megatensors._hub.utils._parsing import parse_duration
from megatensors.hub import MegaHubClient

from ._cli_utils import RepoIdArg, RevisionOpt, TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out
from .jobs_support import key_value_entries
from .download import run_download
from .upload import run_upload


models_cli = typer_factory(help="Manage MEGA model repositories.")
datasets_cli = typer_factory(help="Manage MEGA dataset repositories.")
spaces_cli = typer_factory(help="Manage MEGA space repositories.")
space_secrets_cli = typer_factory(help="Manage write-only Space secrets.")
space_variables_cli = typer_factory(help="Manage Space environment variables.")
space_volumes_cli = typer_factory(help="Manage volumes for a Space.")
spaces_cli.add_group(space_secrets_cli, name="secrets")
spaces_cli.add_group(space_variables_cli, name="variables")
spaces_cli.add_group(space_volumes_cli, name="volumes")


def _visibility(repo: Any) -> str:
    return "private" if repo.private else "public"


def _list_typed_repos(repo_type: str, *, limit: int, token: str | None) -> None:
    repos = MegaHubClient(token=token).list_repos(limit=limit, repo_type=repo_type)
    out.table(
        [
            {
                "repo_id": repo.repo_id,
                "visibility": _visibility(repo),
                "updated_at": repo.updated_at,
            }
            for repo in repos
        ],
        id_key="repo_id",
    )


def _print_typed_repo_info(repo_id: str, repo_type: str, *, token: str | None) -> None:
    if repo_type == "space":
        info = _space_api(token).space_info(repo_id)
        runtime = _space_runtime_record(info.runtime) if info.runtime is not None else None
        out.dict(
            {
                "repo_id": info.id,
                "repo_type": "space",
                "visibility": "private" if info.private else "public",
                "owner": info.author,
                "revision": info.sha,
                "host": info.host,
                "subdomain": info.subdomain,
                "sdk": info.sdk,
                "runtime": runtime,
                "created_at": info.created_at.isoformat() if info.created_at else None,
                "updated_at": info.last_modified.isoformat() if info.last_modified else None,
            },
            id_key="repo_id",
        )
        return
    repo = MegaHubClient(token=token).repo_info(repo_id)
    if repo.repo_type != repo_type:
        out.warning(f"Repository type is {repo.repo_type}, not {repo_type}.")
    out.dict(
        {
            "repo_id": repo.repo_id,
            "repo_type": repo.repo_type,
            "visibility": _visibility(repo),
            "owner": repo.owner,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at,
        },
        id_key="repo_id",
    )


def _list_typed_repo_files(repo_id: str, repo_type: str, *, revision: str, token: str | None) -> None:
    api = MegaHubClient(token=token)
    repo = api.repo_info(repo_id)
    if repo.repo_type != repo_type:
        out.warning(f"Repository type is {repo.repo_type}, not {repo_type}.")
    files = api.list_files(repo_id, revision=revision)
    out.table(
        [{"path": info.path, "size": info.size, "sha256": info.sha256} for info in files],
        id_key="path",
        alignments={"size": "right"},
    )


def _register_typed_repo_commands(group: Any, *, repo_type: str, label: str) -> None:
    """Register one shared repository command set for a MEGA artifact type."""

    if repo_type == "model":
        @group.command(
            "list | ls",
            help="List model repositories, or files in one model repository.",
            examples=["mega models ls --limit 10", "mega models ls owner/model --revision main"],
        )
        def list_repos(
            repo_id: Annotated[str | None, Argument(help="Optional model repository id whose files to list.")] = None,
            limit: Annotated[int, Option("--limit", help="Maximum models to list.", min=1)] = 100,
            revision: RevisionOpt = "main",
            token: TokenOpt = None,
        ) -> None:
            if repo_id is not None:
                _list_typed_repo_files(repo_id, repo_type, revision=revision or "main", token=token)
                return
            if revision not in (None, "main"):
                raise CLIError("--revision requires a model repository id.")
            _list_typed_repos(repo_type, limit=limit, token=token)
    else:
        @group.command("list | ls", help=f"List {label} repositories, or files in a {label} repository.")
        def list_repos(
            repo_id: Annotated[str | None, Argument(help=f"Optional {label} repository id whose files to list.")] = None,
            limit: Annotated[int, Option("--limit", help=f"Maximum {label} repositories to list.", min=1)] = 100,
            revision: RevisionOpt = "main",
            token: TokenOpt = None,
        ) -> None:
            if repo_id is not None:
                _list_typed_repo_files(repo_id, repo_type, revision=revision or "main", token=token)
                return
            _list_typed_repos(repo_type, limit=limit, token=token)

    @group.command("info", help=f"Show {label} repository metadata.")
    def repo_info(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
        _print_typed_repo_info(repo_id, repo_type, token=token)

    if repo_type != "space":
        @group.command("card", help=f"Get the {label} card (README).")
        def repo_card(
            repo_id: RepoIdArg,
            metadata: Annotated[bool, Option("--metadata", help="Output only card metadata.")] = False,
            text: Annotated[bool, Option("--text", help="Output only the card body.")] = False,
            token: TokenOpt = None,
        ) -> None:
            if metadata and text:
                raise CLIError("--metadata and --text are mutually exclusive.")
            card_type = ModelCard if repo_type == "model" else DatasetCard
            card = card_type.load(repo_id, token=token)
            if metadata:
                out.dict(card.data.to_dict())
            elif text:
                out.text(card.text)
            else:
                out.text(card.content)

    @group.command("upload", help=f"Upload a file or folder to a {label} repository.")
    def repo_upload(
        repo_id: RepoIdArg,
        local_path: Annotated[Path, Argument(help="Local file or directory to upload.")] = Path("."),
        path_in_repo: Annotated[
            str | None,
            Argument(help="Remote path. Defaults to the local basename."),
        ] = None,
        revision: RevisionOpt = "main",
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
        include: Annotated[
            list[str] | None,
            Option("--include", help="Glob to include when uploading a folder. Repeatable."),
        ] = None,
        exclude: Annotated[
            list[str] | None,
            Option("--exclude", help="Glob to exclude when uploading a folder. Repeatable."),
        ] = None,
        max_workers: Annotated[
            int | None,
            Option("--max-workers", help="Concurrency for folder uploads.", min=1),
        ] = None,
        sync: Annotated[
            bool,
            Option("--sync", help="Delete remote files missing from the local folder."),
        ] = False,
        token: TokenOpt = None,
    ) -> None:
        run_upload(
            repo_id,
            local_path=local_path,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            private=private,
            commit_message=commit_message,
            commit_description=commit_description,
            include=include,
            exclude=exclude,
            max_workers=max_workers,
            sync=sync,
            token=token,
        )

    @group.command("download", help=f"Download files or a complete snapshot from a {label} repository.")
    def repo_download(
        repo_id: RepoIdArg,
        filenames: Annotated[
            list[str] | None,
            Argument(help="Optional file paths to download. Empty means the full snapshot."),
        ] = None,
        local_dir: Annotated[Path, Option("--local-dir", "-d", help="Output directory.")] = Path("."),
        revision: RevisionOpt = "main",
        include: Annotated[
            list[str] | None,
            Option("--include", help="Glob to include in snapshot mode. Repeatable."),
        ] = None,
        exclude: Annotated[
            list[str] | None,
            Option("--exclude", help="Glob to exclude in snapshot mode. Repeatable."),
        ] = None,
        force: Annotated[
            bool,
            Option("--force", help="Re-download even if the local file exists."),
        ] = False,
        dry_run: Annotated[
            bool,
            Option("--dry-run", help="Show downloads without writing files."),
        ] = False,
        max_workers: Annotated[
            int,
            Option("--max-workers", help="Concurrency for multi-file downloads.", min=1),
        ] = 4,
        token: TokenOpt = None,
    ) -> None:
        run_download(
            repo_id,
            filenames=filenames,
            repo_type=repo_type,
            local_dir=local_dir,
            revision=revision,
            include=include,
            exclude=exclude,
            force=force,
            dry_run=dry_run,
            max_workers=max_workers,
            token=token,
        )


_register_typed_repo_commands(models_cli, repo_type="model", label="model")
_register_typed_repo_commands(datasets_cli, repo_type="dataset", label="dataset")
_register_typed_repo_commands(spaces_cli, repo_type="space", label="space")


@datasets_cli.command(
    "parquet",
    examples=[
        "mega datasets parquet owner/dataset",
        "mega datasets parquet owner/dataset --subset default --split train",
    ],
)
def datasets_parquet(
    dataset_id: Annotated[str, Argument(help="Dataset ID in owner/name form.")],
    subset: Annotated[str | None, Option("--subset", help="Filter Parquet entries by config/subset.")] = None,
    split: Annotated[str | None, Option(help="Filter Parquet entries by split.")] = None,
    token: TokenOpt = None,
) -> None:
    """List Dataset Viewer Parquet files for a dataset."""
    entries = MegaApi(token=token).list_dataset_parquet_files(repo_id=dataset_id, config=subset)
    out.table(
        [
            {"subset": entry.config, "split": entry.split, "url": entry.url, "size": entry.size}
            for entry in entries
            if split is None or entry.split == split
        ],
        headers=["subset", "split", "url", "size"],
        id_key="url",
    )


@datasets_cli.command(
    "sql",
    examples=[
        "mega datasets sql \"SELECT COUNT(*) AS rows FROM read_parquet('https://mega.tensorplay.cn/api/dataset-viewer/outputs/...')\"",
        "mega datasets sql \"SELECT * FROM read_parquet('https://mega.tensorplay.cn/api/dataset-viewer/outputs/...') LIMIT 5\" --format json",
    ],
)
def datasets_sql(
    sql: Annotated[str, Argument(help="Raw SQL query to execute.")],
    token: TokenOpt = None,
) -> None:
    """Execute a local DuckDB query against Dataset Viewer Parquet URLs."""
    try:
        result = execute_raw_sql_query(sql_query=sql, token=token)
    except ImportError as error:
        raise CLIError(str(error)) from error
    out.table(result)


def _space_api(token: str | None) -> MegaApi:
    client = MegaHubClient(token=token)
    return MegaApi(endpoint=client.endpoint, token=client.token)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _space_runtime_record(runtime: Any) -> dict[str, Any]:
    return {
        "stage": _enum_value(runtime.stage),
        "hardware": _enum_value(runtime.hardware),
        "requested_hardware": _enum_value(runtime.requested_hardware),
        "sleep_time": runtime.sleep_time,
        "storage": _enum_value(runtime.storage),
        "dev_mode": bool(runtime.dev_mode),
        "volumes": [volume.to_dict() for volume in runtime.volumes or []],
        "generation": runtime.raw.get("generation"),
        "updated_at": runtime.raw.get("updatedAt"),
    }


@spaces_cli.command("runtime", examples=["mega spaces runtime username/my-space"])
def spaces_runtime(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """Show the live runtime state of a Space."""
    out.dict(_space_runtime_record(_space_api(token).get_space_runtime(repo_id)), id_key="stage")


@spaces_cli.command("wait", examples=["mega spaces wait username/my-space --timeout 5m"])
def spaces_wait(
    repo_id: RepoIdArg,
    timeout: Annotated[str | None, Option("--timeout", help="Maximum wait, such as 300s, 5m, or 1h.")] = None,
    interval: Annotated[float, Option("--interval", help="Polling interval in seconds.")] = 1.0,
    token: TokenOpt = None,
) -> None:
    """Wait until a Space leaves its build or startup stage."""
    try:
        timeout_seconds = parse_duration(timeout) if timeout is not None else None
    except ValueError as error:
        raise CLIError(str(error)) from error
    runtime = _space_api(token).wait_for_space(repo_id, timeout=timeout_seconds, poll_interval=interval)
    out.dict(_space_runtime_record(runtime), id_key="stage")
    if _enum_value(runtime.stage) != "RUNNING":
        raise CLIError(f"Space '{repo_id}' settled in stage '{_enum_value(runtime.stage)}'.")


@spaces_cli.command("pause", examples=["mega spaces pause username/my-space"])
def spaces_pause(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """Pause a Space so its runtime and usage billing stop."""
    runtime = _space_api(token).pause_space(repo_id)
    out.result("Space paused", space_id=repo_id, stage=_enum_value(runtime.stage))


@spaces_cli.command(
    "restart",
    examples=["mega spaces restart username/my-space", "mega spaces restart username/my-space --factory-reboot"],
)
def spaces_restart(
    repo_id: RepoIdArg,
    factory_reboot: Annotated[
        bool,
        Option("--factory-reboot", help="Rebuild from scratch without using the build cache."),
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Restart a Space and resume its runtime."""
    runtime = _space_api(token).restart_space(repo_id, factory_reboot=factory_reboot)
    out.result(
        "Space restart triggered",
        space_id=repo_id,
        stage=_enum_value(runtime.stage),
        factory_reboot=factory_reboot,
    )


@spaces_cli.command("hardware", examples=["mega spaces hardware"])
def spaces_hardware(token: TokenOpt = None) -> None:
    """List the live hardware options available for MEGA Spaces."""
    items = []
    for hardware in _space_api(token).list_spaces_hardware():
        accelerator = getattr(hardware, "accelerator", None)
        accelerator_description = (
            f"{accelerator.quantity}x {accelerator.model} ({accelerator.vram})" if accelerator else None
        )
        cost_minute = f"${hardware.unit_cost_usd:.4f}" if hardware.unit_cost_usd else "free"
        cost_hour = f"${hardware.unit_cost_usd * 60:.2f}" if hardware.unit_cost_usd else "free"
        items.append({
            "name": hardware.name,
            "pretty name": hardware.pretty_name,
            "cpu": hardware.cpu,
            "ram": hardware.ram,
            "accelerator": accelerator_description,
            "cost/min": cost_minute,
            "cost/hour": cost_hour,
        })
    out.table(items, id_key="name")
    out.hint("Use `mega spaces settings <space_id> --hardware <name>` to request hardware for a Space.")


@spaces_cli.command(
    "settings",
    examples=[
        "mega spaces settings username/my-space --hardware cpu-upgrade",
        "mega spaces settings username/my-space --sleep-time 15m",
    ],
)
def spaces_settings(
    repo_id: RepoIdArg,
    hardware: Annotated[
        str | None,
        Option("--hardware", help="Space hardware flavor. Run `mega spaces hardware` to list currently available options."),
    ] = None,
    sleep_time: Annotated[
        int | None,
        Option("--sleep-time", help="Idle seconds before sleep. Use -1 to never sleep; availability depends on the selected hardware."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Update Space hardware and sleep settings."""
    if hardware is None and sleep_time is None:
        raise CLIError("Specify at least one setting to update.")
    api = _space_api(token)
    runtime = (
        api.request_space_hardware(repo_id, hardware=hardware, sleep_time=sleep_time)
        if hardware is not None
        else api.set_space_sleep_time(repo_id, sleep_time=sleep_time)
    )
    out.result(
        "Space settings updated",
        space_id=repo_id,
        hardware=_enum_value(runtime.requested_hardware),
        sleep_time=runtime.sleep_time,
    )
    out.hint(f"Use `mega spaces info {repo_id}` to verify the runtime configuration.")


@spaces_cli.command("dev-mode", examples=["mega spaces dev-mode username/my-space", "mega spaces dev-mode username/my-space --stop"])
def spaces_dev_mode(
    repo_id: RepoIdArg,
    stop: Annotated[bool, Option("--stop", help="Disable Dev Mode.")] = False,
    token: TokenOpt = None,
) -> None:
    """Enable or disable the plan-gated Space Dev Mode setting."""
    api = _space_api(token)
    runtime = api.disable_space_dev_mode(repo_id) if stop else api.enable_space_dev_mode(repo_id)
    out.dict(_space_runtime_record(runtime), id_key="stage")


@spaces_cli.command("logs", examples=["mega spaces logs username/my-space", "mega spaces logs --build -n 100 username/my-space"])
def spaces_logs(
    repo_id: RepoIdArg,
    build: Annotated[bool, Option("--build", help="Show build logs instead of application logs.")] = False,
    follow: Annotated[bool, Option("-f", "--follow", help="Stream new log lines.")] = False,
    tail: Annotated[int | None, Option("-n", "--tail", help="Show only the last N lines.", min=1)] = None,
    token: TokenOpt = None,
) -> None:
    """Print retained Space build or application logs."""
    if follow and tail is not None:
        raise CLIError("--follow and --tail cannot be used together.")
    logs: Any = _space_api(token).fetch_space_logs(repo_id, build=build, follow=follow)
    if tail is not None:
        logs = deque(logs, maxlen=tail)
    for line in logs:
        out.text(str(line).rstrip("\n"))


def _env_entries(values: list[str] | None, file: Path | None, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values or []:
        key, separator, value = raw.partition("=")
        if separator:
            result.update(key_value_entries([raw], label))
        else:
            if key not in os.environ:
                raise CLIError(f"Local environment variable '{key}' is not set.")
            result.update(key_value_entries([f"{key}={os.environ[key]}"], label))
    if file is not None:
        if not file.is_file():
            raise CLIError(f"Environment file '{file}' does not exist.")
        result.update(load_dotenv(file.read_text(), environ=os.environ))
    return result


@space_secrets_cli.command("list | ls", examples=["mega spaces secrets list username/my-space"])
def space_secrets_list(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """List secret metadata without exposing values."""
    secrets = _space_api(token).get_space_secrets(repo_id)
    out.table([
        {
            "key": secret.key,
            "description": secret.description,
            "updated_at": secret.updated_at.isoformat() if secret.updated_at else None,
        }
        for secret in secrets.values()
    ], id_key="key")


@space_secrets_cli.command("add", examples=["mega spaces secrets add username/my-space -s API_TOKEN"])
def space_secrets_add(
    repo_id: RepoIdArg,
    secrets: Annotated[list[str] | None, Option("-s", "--secrets", help="KEY=VALUE or a local environment key. Repeatable.")] = None,
    secrets_file: Annotated[Path | None, Option("--secrets-file", help="Read secrets from a dotenv file.")] = None,
    token: TokenOpt = None,
) -> None:
    """Add or replace one or more write-only Space secrets."""
    entries = _env_entries(secrets, secrets_file, "Secret")
    if not entries:
        raise CLIError("Specify at least one -s/--secrets entry or --secrets-file.")
    api = _space_api(token)
    for key, value in entries.items():
        api.add_space_secret(repo_id, key=key, value=value)
    out.result("Space secrets saved", space_id=repo_id, keys=sorted(entries))


@space_secrets_cli.command("delete", examples=["mega spaces secrets delete username/my-space API_TOKEN --yes"])
def space_secrets_delete(
    repo_id: RepoIdArg,
    key: Annotated[str, Argument(help="Secret key to delete.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a Space secret."""
    out.confirm(f"Delete secret '{key}' from '{repo_id}'?", yes=yes)
    _space_api(token).delete_space_secret(repo_id, key=key)
    out.result("Space secret deleted", space_id=repo_id, key=key)


@space_variables_cli.command("list | ls", examples=["mega spaces variables list username/my-space"])
def space_variables_list(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """List Space environment variables."""
    variables = _space_api(token).get_space_variables(repo_id)
    out.table([
        {
            "key": variable.key,
            "value": variable.value,
            "description": variable.description,
            "updated_at": variable.updated_at.isoformat() if variable.updated_at else None,
        }
        for variable in variables.values()
    ], id_key="key")


@space_variables_cli.command("add", examples=["mega spaces variables add username/my-space -e MODE=production"])
def space_variables_add(
    repo_id: RepoIdArg,
    env: Annotated[list[str] | None, Option("-e", "--env", help="KEY=VALUE. Repeatable.")] = None,
    env_file: Annotated[Path | None, Option("--env-file", help="Read variables from a dotenv file.")] = None,
    token: TokenOpt = None,
) -> None:
    """Add or replace one or more Space environment variables."""
    entries = _env_entries(env, env_file, "Variable")
    if not entries:
        raise CLIError("Specify at least one -e/--env entry.")
    api = _space_api(token)
    for key, value in entries.items():
        api.add_space_variable(repo_id, key=key, value=value)
    out.result("Space variables saved", space_id=repo_id, keys=sorted(entries))


@space_variables_cli.command("delete", examples=["mega spaces variables delete username/my-space MODE --yes"])
def space_variables_delete(
    repo_id: RepoIdArg,
    key: Annotated[str, Argument(help="Variable key to delete.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip confirmation.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a Space environment variable."""
    out.confirm(f"Delete variable '{key}' from '{repo_id}'?", yes=yes)
    _space_api(token).delete_space_variable(repo_id, key=key)
    out.result("Space variable deleted", space_id=repo_id, key=key)


@spaces_cli.command("card", examples=["mega spaces card username/my-space", "mega spaces card username/my-space --metadata"])
def spaces_card(
    repo_id: RepoIdArg,
    metadata: Annotated[bool, Option("--metadata", help="Output only card metadata.")] = False,
    text: Annotated[bool, Option("--text", help="Output only the card body.")] = False,
    token: TokenOpt = None,
) -> None:
    """Get the Space card (README) for a Space."""
    if metadata and text:
        raise CLIError("--metadata and --text are mutually exclusive.")
    card = SpaceCard.load(repo_id, token=token)
    if metadata:
        out.dict(card.data.to_dict())
    elif text:
        out.text(card.text)
    else:
        out.text(card.content)


@spaces_cli.command("templates", examples=["mega spaces templates"])
def spaces_templates(token: TokenOpt = None) -> None:
    """List the available Space templates."""
    templates = _space_api(token).list_space_templates()
    out.table(
        [
            {
                "name": template.name,
                "repo_id": template.repo_id,
                "sdk": template.sdk,
                "preferred_private": template.preferred_private,
            }
            for template in templates
        ],
        id_key="name",
    )


@spaces_cli.command("search", examples=['mega spaces search "generate image"', 'mega spaces search "chatbot" --sdk gradio'])
def spaces_search(
    query: Annotated[str, Argument(help="Search query.")],
    filter: Annotated[list[str] | None, Option("--filter", help="Tag filter. Repeatable.")] = None,
    sdk: Annotated[list[str] | None, Option("--sdk", help="SDK filter. Repeatable.")] = None,
    include_non_running: Annotated[bool, Option("--include-non-running", help="Include non-running Spaces.")] = False,
    description: Annotated[bool, Option("--description", help="Show generated descriptions.")] = False,
    limit: Annotated[int, Option("--limit", help="Maximum results.", min=1)] = 10,
    token: TokenOpt = None,
) -> None:
    """Search Spaces using the Hub search service."""
    results = itertools.islice(
        _space_api(token).search_spaces(
            query,
            filter=filter,
            sdk=sdk,
            include_non_running=include_non_running,
        ),
        limit,
    )
    rows = []
    for result in results:
        runtime = getattr(result, "runtime", None)
        row = {
            "id": result.id,
            "title": getattr(result, "title", None),
            "sdk": getattr(result, "sdk", None),
            "likes": getattr(result, "likes", None),
            "stage": getattr(runtime, "stage", None),
            "score": getattr(result, "semantic_relevancy_score", None),
        }
        if description:
            row["description"] = getattr(result, "ai_short_description", None)
        rows.append(row)
    out.table(rows, id_key="id")


def _parse_space_volumes(values: list[str] | None) -> list[Volume]:
    volumes: list[Volume] = []
    for value in values or []:
        try:
            mount = parse_mega_mount(value)
        except ValueError as error:
            raise CLIError(str(error)) from error
        volumes.append(
            Volume(
                type=str(mount.source.type),
                source=mount.source.id,
                mount_path=mount.mount_path,
                revision=mount.source.revision,
                read_only=mount.read_only,
                path=mount.source.path_in_repo or None,
            )
        )
    return volumes


@space_volumes_cli.command("list | ls", examples=["mega spaces volumes ls username/my-space"])
def space_volumes_list(repo_id: RepoIdArg, token: TokenOpt = None) -> None:
    """List volumes mounted in a Space."""
    runtime = _space_api(token).get_space_runtime(repo_id)
    out.table([volume.to_dict() for volume in runtime.volumes or []], id_key="mountPath")


@space_volumes_cli.command("set", examples=["mega spaces volumes set username/my-space -v mega://datasets/user/data:/datasets/data:ro"])
def space_volumes_set(
    repo_id: RepoIdArg,
    volume: Annotated[list[str] | None, Option("-v", "--volume", help="MEGA mount URI. Repeatable.")] = None,
    token: TokenOpt = None,
) -> None:
    """Replace all volumes mounted in a Space."""
    volumes = _parse_space_volumes(volume)
    if not volumes:
        raise CLIError("At least one volume must be specified with -v/--volume.")
    _space_api(token).set_space_volumes(repo_id, volumes=volumes)
    out.result("Volumes set", space_id=repo_id, volumes=[volume.to_uri() for volume in volumes])


@space_volumes_cli.command("delete", examples=["mega spaces volumes delete username/my-space --yes"])
def space_volumes_delete(
    repo_id: RepoIdArg,
    yes: Annotated[bool, Option("-y", "--yes", help="Skip confirmation.")] = False,
    token: TokenOpt = None,
) -> None:
    """Remove every volume from a Space."""
    out.confirm(f"Remove all volumes from Space '{repo_id}'?", yes=yes)
    _space_api(token).delete_space_volumes(repo_id)
    out.result("Volumes deleted", space_id=repo_id)


@spaces_cli.command(
    "ssh",
    examples=["mega spaces ssh username/my-space", "mega spaces ssh username/my-space --dry-run"],
    context_settings={"ignore_unknown_options": True},
)
def spaces_ssh(
    repo_id: RepoIdArg,
    remote_command: Annotated[list[str] | None, Argument(help="Optional command to run instead of an interactive shell.")] = None,
    identity_file: Annotated[Path | None, Option("-i", "--identity-file", help="SSH private key.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", help="Print the SSH command without running it.")] = False,
    auto: Annotated[bool, Option("--auto", help="Enable Dev Mode without prompting.")] = False,
    token: TokenOpt = None,
) -> None:
    """SSH into a Space Dev Mode container."""
    # Click stops option parsing once it reaches the variadic remote command.
    # Preserve the documented ``spaces ssh REPO --dry-run`` form by consuming
    # only recognized control options before the first remote argv item.
    remote_args = list(remote_command or [])
    while remote_args:
        if remote_args[0] == "--dry-run":
            dry_run = True
            remote_args.pop(0)
            continue
        if remote_args[0] == "--auto":
            auto = True
            remote_args.pop(0)
            continue
        if remote_args[0] in {"-i", "--identity-file"} and len(remote_args) >= 2:
            identity_file = Path(remote_args[1])
            del remote_args[:2]
            continue
        break
    api = _space_api(token)
    info = api.space_info(repo_id)
    if info.runtime is None or not info.runtime.dev_mode:
        out.confirm(f"Dev Mode is disabled on '{repo_id}'. Enable it now?", yes=auto)
        api.enable_space_dev_mode(repo_id)
        runtime = api.wait_for_space(repo_id)
        if _enum_value(runtime.stage) != "RUNNING":
            raise CLIError(f"Space '{repo_id}' is not running (stage='{_enum_value(runtime.stage)}').")
        info = api.space_info(repo_id)
    if not info.subdomain:
        raise CLIError(f"Space '{repo_id}' has no running Dev Mode endpoint yet.")
    cloudflared = shutil.which("cloudflared") or "cloudflared"
    host = os.environ.get("MEGA_SPACE_SSH_HOST", "ssh.mega.space")
    if not all(part and part.replace("-", "").isalnum() for part in host.split(".")):
        raise CLIError("MEGA_SPACE_SSH_HOST must be a valid hostname.")
    request = base64.urlsafe_b64encode(json.dumps({
        "repo_id": repo_id,
        "command": remote_args,
    }, separators=(",", ":")).encode()).decode().rstrip("=")
    command = [
        "ssh",
        "-o", f"ProxyCommand={cloudflared} access ssh --hostname %h",
        "-o", "RequestTTY=force",
    ]
    if identity_file is not None:
        command.extend(["-i", str(identity_file.expanduser())])
    command.extend([f"mega-spaces@{host}", request])
    if dry_run:
        click.echo(shlex.join(command))
        return
    raise SystemExit(subprocess.call(command))


@spaces_cli.command("hot-reload", examples=["mega spaces hot-reload username/my-space app.py -f app.py"])
def spaces_hot_reload(
    repo_id: RepoIdArg,
    filename: Annotated[str | None, Argument(help="Python path in the Space repository.")] = None,
    local_file: Annotated[Path | None, Option("-f", "--local-file", help="Local file to upload.")] = None,
    skip_checks: Annotated[bool, Option("--skip-checks", help="Skip SDK compatibility checks.")] = False,
    skip_summary: Annotated[bool, Option("--skip-summary", help="Skip the result summary.")] = False,
    token: TokenOpt = None,
) -> None:
    """Commit a Python file through the Space hot-reload compatibility flow."""
    api = _space_api(token)
    info = api.space_info(repo_id)
    if not skip_checks and info.sdk != "gradio":
        raise CLIError(f"Hot reload is unavailable for Space SDK '{info.sdk}'.")
    if local_file is not None:
        source = local_file
        filename = filename or local_file.as_posix()
    elif filename is not None:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise CLIError("Interactive editing requires a TTY; use -f/--local-file.")
        editor = os.environ.get("MEGA_EDITOR") or os.environ.get("VISUAL") or os.environ.get("EDITOR")
        editor = editor or next((name for name in ("code --wait", "nvim", "nano", "vim", "vi") if shutil.which(name.split()[0])), None)
        if editor is None:
            raise CLIError("No editor found; use -f/--local-file.")
        temporary = tempfile.TemporaryDirectory()
        try:
            source = Path(mega_hub_download(repo_id, filename, repo_type="space", local_dir=temporary.name, token=token))
        except Exception:
            source = Path(temporary.name) / filename
            source.parent.mkdir(parents=True, exist_ok=True)
            source.touch()
        if subprocess.call([*shlex.split(editor), str(source)]) != 0:
            raise CLIError("Editor exited with an error.")
    else:
        raise CLIError("Either filename or --local-file/-f must be specified.")
    if not source.is_file():
        raise CLIError(f"Local file '{source}' does not exist.")
    commit = api.upload_file(
        repo_type="space",
        repo_id=repo_id,
        path_or_fileobj=source,
        path_in_repo=filename,
        parent_commit=None if skip_checks else info.sha,
        _hot_reload=True,
    )
    if not skip_summary:
        out.result("Space hot reload triggered", space_id=repo_id, path=filename, commit=commit.oid)

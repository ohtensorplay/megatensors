"""Discover and safely install MEGA MCP marketplace companions."""

import json
import re
import tempfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Callable, TypeVar

from megatensors._hub.errors import CLIError
from megatensors.hub import McpMarketplaceInfo, MegaHubClient, MegaHubError

from ._cli_utils import RevisionOpt, TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import OutputFormat, out


MCP_INSTALL_ROOT = Path("~/.local/share/mega/mcp")
MANAGED_MARKER = ".mega-mcp.json"
MCP_COMPANION_EXCLUDES = (".git*", "**/.git*")
_REPO_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
F = TypeVar("F", bound=Callable[..., Any])

mcp_cli = typer_factory(
    help="Discover MCPs and install their versioned local companion files."
)


def _marketplace_errors_as_cli_errors(command: F) -> F:
    @wraps(command)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except MegaHubError as error:
            raise CLIError(str(error)) from error
        except ValueError as error:
            raise CLIError(str(error)) from error

    return wrapped  # type: ignore[return-value]


def _safe_repo_id(repo_id: str) -> tuple[str, str]:
    parts = repo_id.split("/")
    if (
        len(parts) != 2
        or any(part in {".", ".."} for part in parts)
        or not all(_REPO_SEGMENT.fullmatch(part) for part in parts)
    ):
        raise CLIError("MCP id must use the namespace/name form.")
    return parts[0], parts[1]


def _listing_record(listing: McpMarketplaceInfo) -> dict[str, Any]:
    return {
        "repo_id": listing.repo_id,
        "title": listing.title,
        "summary": listing.summary,
        "category": listing.category,
        "price_per_call": listing.price_per_call,
        "runtime": listing.runtime_kind,
        "available": listing.runtime_available,
    }


def _install_companion(
    repo_id: str,
    *,
    root: Path,
    revision: str,
    force: bool,
    token: str | None,
    max_workers: int,
) -> Path:
    namespace, name = _safe_repo_id(repo_id)
    install_root = root.expanduser().resolve()
    owner_root = install_root / namespace
    owner_root.mkdir(parents=True, exist_ok=True)
    destination = owner_root / name
    if destination.exists() and not force:
        raise CLIError(
            f"MCP companion already exists: {destination}. Use --force to overwrite it."
        )

    with tempfile.TemporaryDirectory(
        dir=owner_root, prefix=f".{name}.install-"
    ) as temporary:
        temporary_root = Path(temporary)
        staged = temporary_root / name
        MegaHubClient(token=token).snapshot_download(
            repo_id,
            local_dir=staged,
            revision=revision,
            exclude=MCP_COMPANION_EXCLUDES,
            force=True,
            max_workers=max_workers,
        )
        if not (staged / "README.md").is_file():
            raise CLIError(f"MCP companion '{repo_id}' is missing README.md.")
        marker = {
            "schema": 1,
            "repo_id": repo_id,
            "revision": revision,
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        (staged / MANAGED_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if destination.exists():
            backup = temporary_root / f"{name}.backup"
            destination.rename(backup)
            try:
                staged.rename(destination)
            except Exception:
                backup.rename(destination)
                raise
        else:
            staged.rename(destination)
    return destination


@mcp_cli.command(
    "search",
    examples=[
        "mega mcp search",
        "mega mcp search xpuoj",
        "mega mcp search judge --category developer-tools --format json",
    ],
)
@_marketplace_errors_as_cli_errors
def mcp_search(
    query: Annotated[
        str | None, Argument(help="Optional text to search in MCP ids and descriptions.")
    ] = None,
    category: Annotated[
        str | None, Option("--category", help="Filter by marketplace category.")
    ] = None,
    limit: Annotated[
        int, Option("--limit", help="Maximum MCPs to return (up to 80).", min=1)
    ] = 80,
    token: TokenOpt = None,
) -> None:
    """Search the public MEGA MCP marketplace."""
    listings = MegaHubClient(token=token).list_mcp_marketplace(
        search=query, category=category, limit=limit
    )
    out.table([_listing_record(listing) for listing in listings], id_key="repo_id")


@mcp_cli.command("info", examples=["mega mcp info mega/xpuoj"])
@_marketplace_errors_as_cli_errors
def mcp_info(
    repo_id: Annotated[str, Argument(help="MCP id in namespace/name form.")],
    token: TokenOpt = None,
) -> None:
    """Show one MCP marketplace entry."""
    _safe_repo_id(repo_id)
    listing = MegaHubClient(token=token).get_mcp_marketplace(repo_id)
    out.dict(
        {
            **_listing_record(listing),
            "description": listing.description,
            "tags": list(listing.tags),
            "owner": {
                "handle": listing.owner_handle,
                "display_name": listing.owner_display_name,
            },
            "endpoint": listing.endpoint,
            "url": listing.url,
            "runtime_sdk": listing.runtime_sdk,
            "runtime_stage": listing.runtime_stage,
            "calls": listing.calls,
            "consumers": listing.consumers,
            "earned": listing.earned,
            "published_at": listing.published_at,
        },
        id_key="repo_id",
    )


@mcp_cli.command(
    "install",
    examples=[
        "mega mcp install mega/xpuoj",
        "mega mcp install mega/xpuoj --dest ./.mega/mcp",
        "mega mcp install mega/xpuoj --revision v0.1.1 --force",
    ],
)
@_marketplace_errors_as_cli_errors
def mcp_install(
    repo_id: Annotated[str, Argument(help="MCP id in namespace/name form.")],
    dest: Annotated[
        Path,
        Option(
            "--dest",
            "-d",
            help="Installation root. MCPs are stored under namespace/name.",
        ),
    ] = MCP_INSTALL_ROOT,
    revision: RevisionOpt = "main",
    force: Annotated[
        bool, Option("--force", help="Atomically replace an existing installation.")
    ] = False,
    max_workers: Annotated[
        int, Option("--max-workers", help="Concurrent file downloads.", min=1)
    ] = 4,
    token: TokenOpt = None,
) -> None:
    """Install versioned MCP companion files without executing publisher code."""
    _safe_repo_id(repo_id)
    listing = MegaHubClient(token=token).get_mcp_marketplace(repo_id)
    if not listing.runtime_available:
        out.warning("The remote MCP runtime is currently unavailable.")
    status = out.status(f"Installing {repo_id} companion files...")
    destination = _install_companion(
        repo_id,
        root=dest,
        revision=revision or "main",
        force=force,
        token=token,
        max_workers=max_workers,
    )
    status.done(f"Installed {repo_id}.")
    out.result(
        "MCP companion installed",
        repo_id=repo_id,
        path=destination,
        readme=destination / "README.md",
        executed=False,
    )
    if out.mode == OutputFormat.human:
        out.hint(
            "Review README.md before enabling any included CLI, skill, or plugin."
        )

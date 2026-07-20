"""Generate, install, and update skills for AI coding assistants."""

import json
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated

from click import Command, Context, Group

from megatensors import __version__
from megatensors.hub import MegaHubClient
from megatensors._hub._buckets import BucketFile
from megatensors._hub.errors import CLIError
from megatensors._hub.mega_api import MegaApi

from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out


DEFAULT_SKILL_ID = "mega-cli"
DEFAULT_SKILLS_BUCKET_ID = "ohtensorplay/skills"
MARKETPLACE_PATH = "marketplace.json"
MANAGED_MARKER = ".mega-skill-manifest.json"
CENTRAL_LOCAL = Path(".agents/skills")
CENTRAL_GLOBAL = Path("~/.agents/skills")
CLAUDE_LOCAL = Path(".claude/skills")
CLAUDE_GLOBAL = Path("~/.claude/skills")
skills_cli = typer_factory(help="Manage skills for AI assistants.")


def _api(token: str | None = None) -> MegaApi:
    client = MegaHubClient(token=token)
    return MegaApi(endpoint=client.endpoint, token=client.token)


def _collect_leaves(group: Group, ctx: Context, path: list[str]) -> list[tuple[list[str], Command]]:
    leaves: list[tuple[list[str], Command]] = []
    child_context = Context(group, parent=ctx, info_name=path[-1] if path else "mega")
    for name in group.list_commands(child_context):
        command = group.get_command(child_context, name)
        if command is None or command.hidden:
            continue
        command_path = [*path, name]
        if isinstance(command, Group):
            leaves.extend(_collect_leaves(command, child_context, command_path))
        else:
            leaves.append((command_path, command))
    return leaves


def _required_params(command: Command) -> str:
    values: list[str] = []
    for parameter in command.params:
        if not parameter.required or parameter.name == "help":
            continue
        long_name = next((value for value in getattr(parameter, "opts", ()) if value.startswith("--")), None)
        if long_name:
            values.append(f"{long_name} {getattr(parameter.type, 'name', 'VALUE').upper()}")
        else:
            values.append(parameter.human_readable_name.upper())
    return " ".join(values)


def build_skill_md() -> str:
    """Generate a compact, versioned command reference from the live command tree."""
    from .mega import app

    context = Context(app, info_name="mega")
    lines = [
        "---",
        "name: mega-cli",
        'description: "Use the MEGA Hub CLI to manage models, datasets, Spaces, buckets, Jobs, sandboxes, repositories, inference, extensions, and skills."',
        "---",
        "",
        "The `mega` command is available. Authenticate with `mega auth login` or `MEGA_TOKEN`.",
        f"Generated with `megatensors v{__version__}`. Regenerate with `mega skills add --force`.",
        "",
        "## Commands",
        "",
    ]
    for path, command in _collect_leaves(app, context, []):
        required = _required_params(command)
        invocation = " ".join(["mega", *path, *([required] if required else [])])
        summary = (command.help or "").splitlines()[0].strip()
        lines.append(f"- `{invocation}` — {summary}")
    lines.extend(
        [
            "",
            "## Usage notes",
            "",
            "- Run `mega <command> --help` for complete options and examples.",
            "- Prefer `MEGA_TOKEN` over passing `--token` on the command line.",
            "- Use `--format json` for automation and `--quiet` for identifiers only.",
            "",
        ]
    )
    return "\n".join(lines)


def _marketplace(token: str | None = None) -> list[dict[str, str]]:
    api = _api(token)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / MARKETPLACE_PATH
        api.download_bucket_files(DEFAULT_SKILLS_BUCKET_ID, [(MARKETPLACE_PATH, path)], raise_on_missing_files=True)
        payload = json.loads(path.read_text())
    plugins = payload.get("plugins") if isinstance(payload, dict) else None
    if not isinstance(plugins, list):
        raise CLIError("Invalid skill marketplace: expected a top-level 'plugins' list.")
    result = []
    for plugin in plugins:
        if not isinstance(plugin, dict) or not isinstance(plugin.get("name"), str) or not isinstance(plugin.get("source"), str):
            continue
        result.append({
            "name": _safe_name(plugin["name"]),
            "source": _safe_marketplace_source(plugin["source"]),
            "description": str(plugin.get("description") or ""),
        })
    return result


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise CLIError(f"Invalid skill name '{name}'.")
    return name


def _safe_marketplace_source(source: str) -> str:
    path = PurePosixPath(source)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise CLIError(f"Invalid marketplace source '{source}'.")
    return path.as_posix()


def _atomic_install(name: str, root: Path, populate, *, force: bool) -> Path:
    name = _safe_name(name)
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.exists() and not force:
        raise CLIError(f"Skill already exists: {destination}. Use --force to overwrite it.")
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{name}.install-") as temporary:
        staged = Path(temporary) / name
        populate(staged)
        if not (staged / "SKILL.md").is_file():
            raise CLIError(f"Skill '{name}' is missing SKILL.md.")
        (staged / MANAGED_MARKER).touch()
        if destination.exists():
            backup = Path(temporary) / f"{name}.backup"
            destination.rename(backup)
            try:
                staged.rename(destination)
            except Exception:
                backup.rename(destination)
                raise
        else:
            staged.rename(destination)
    return destination


def _install_generated(root: Path, *, force: bool) -> Path:
    def populate(staged: Path) -> None:
        staged.mkdir(parents=True)
        (staged / "SKILL.md").write_text(build_skill_md())

    return _atomic_install(DEFAULT_SKILL_ID, root, populate, force=force)


def _install_marketplace(name: str, root: Path, *, force: bool, token: str | None = None) -> Path:
    entry = next((item for item in _marketplace(token) if item["name"].lower() == name.lower()), None)
    if entry is None:
        raise CLIError(f"Skill '{name}' was not found in {DEFAULT_SKILLS_BUCKET_ID}.")
    prefix = entry["source"].rstrip("/")
    api = _api(token)

    def populate(staged: Path) -> None:
        staged.mkdir(parents=True)
        files = [item for item in api.list_bucket_tree(DEFAULT_SKILLS_BUCKET_ID, prefix=prefix, recursive=True) if isinstance(item, BucketFile)]
        specs = []
        for bucket_file in files:
            marker = f"{prefix}/"
            if not bucket_file.path.startswith(marker):
                continue
            relative = PurePosixPath(bucket_file.path[len(marker):])
            if not relative.parts or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise CLIError(f"Marketplace file escapes skill directory: '{bucket_file.path}'.")
            local = staged.joinpath(*relative.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            specs.append((bucket_file, local))
        if not specs:
            raise CLIError(f"Marketplace path '{prefix}' contains no files.")
        api.download_bucket_files(DEFAULT_SKILLS_BUCKET_ID, specs)

    return _atomic_install(entry["name"], root, populate, force=force)


def _install(name: str, root: Path, *, force: bool, token: str | None = None) -> Path:
    return _install_generated(root, force=force) if name == DEFAULT_SKILL_ID else _install_marketplace(name, root, force=force, token=token)


def _link(root: Path, name: str, target: Path, *, force: bool) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    link = root / name
    if link.exists() or link.is_symlink():
        if not force:
            raise CLIError(f"Skill link already exists: {link}. Use --force to overwrite it.")
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target, target_is_directory=True)
    return link


def _skill_dirs(roots: list[Path]) -> list[Path]:
    found: dict[Path, Path] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_dir():
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    found.setdefault(child.resolve(), child)
    return sorted(found.values(), key=lambda path: path.name)


@skills_cli.command("preview")
def skills_preview() -> None:
    """Print the generated mega-cli SKILL.md."""
    print(build_skill_md())


@skills_cli.command("list | ls", examples=["mega skills list"])
def skills_list(token: TokenOpt = None) -> None:
    """List marketplace skills and local installation locations."""
    locations = [("project", CENTRAL_LOCAL), ("project_claude", CLAUDE_LOCAL), ("global", CENTRAL_GLOBAL), ("global_claude", CLAUDE_GLOBAL)]
    installed: dict[str, set[str]] = {}
    for label, root in locations:
        for directory in _skill_dirs([root]):
            installed.setdefault(directory.name.lower(), set()).add(label)
    rows = []
    for skill in _marketplace(token):
        row = {"name": skill["name"], "description": skill["description"]}
        row.update({label: label in installed.get(skill["name"].lower(), set()) for label, _ in locations})
        rows.append(row)
    out.table(rows, id_key="name")


@skills_cli.command("add", examples=["mega skills add", "mega skills add mega-spaces --global"])
def skills_add(
    name: Annotated[str, Argument(help="Marketplace skill name.")] = DEFAULT_SKILL_ID,
    claude: Annotated[bool, Option("--claude", help="Also link into Claude's skills directory.")] = False,
    global_: Annotated[bool, Option("--global", "-g", help="Install at user level.")] = False,
    dest: Annotated[Path | None, Option("--dest", help="Custom skills directory.")] = None,
    force: Annotated[bool, Option("--force", help="Overwrite an existing managed skill.")] = False,
    token: TokenOpt = None,
) -> None:
    """Install the generated mega-cli skill or a marketplace skill."""
    if dest is not None and (claude or global_):
        raise CLIError("--dest cannot be combined with --claude or --global.")
    central = dest if dest is not None else (CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL)
    installed = _install(name, central, force=force, token=token)
    out.result("Skill installed", name=name, path=str(installed))
    if claude:
        link = _link(CLAUDE_GLOBAL if global_ else CLAUDE_LOCAL, name, installed, force=force)
        out.result("Claude skill linked", name=name, path=str(link))


@skills_cli.command("update", examples=["mega skills update", "mega skills update mega-cli"])
def skills_update(
    name: Annotated[str | None, Argument(help="Optional installed skill name.")] = None,
    claude: Annotated[bool, Option("--claude", help="Include Claude's skills directory.")] = False,
    global_: Annotated[bool, Option("--global", "-g", help="Use user-level directories.")] = False,
    dest: Annotated[Path | None, Option("--dest", help="Custom skills directory.")] = None,
    token: TokenOpt = None,
) -> None:
    """Refresh managed skills from the current CLI or marketplace bucket."""
    if dest is not None and (claude or global_):
        raise CLIError("--dest cannot be combined with --claude or --global.")
    roots = [dest] if dest is not None else [CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL]
    if claude:
        roots.append(CLAUDE_GLOBAL if global_ else CLAUDE_LOCAL)
    directories = [directory for directory in _skill_dirs([root for root in roots if root is not None]) if name is None or directory.name.lower() == name.lower()]
    if name is not None and not directories:
        raise CLIError(f"No installed skill matches '{name}'.")
    rows = []
    for directory in directories:
        if not (directory / MANAGED_MARKER).exists():
            rows.append({"name": directory.name, "status": "unmanaged", "path": str(directory)})
            continue
        try:
            _install(directory.name, directory.parent, force=True, token=token)
            rows.append({"name": directory.name, "status": "updated", "path": str(directory)})
        except Exception as error:
            rows.append({"name": directory.name, "status": "source_unreachable", "detail": str(error), "path": str(directory)})
    out.table(rows, id_key="name")

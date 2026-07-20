"""Install and execute trusted third-party extensions for the ``mega`` CLI."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import click

from megatensors._hub.errors import CLIError
from megatensors._hub.utils import get_session, mega_raise_for_status

from ._cli_utils import MegaCliGroup, typer_factory
from ._framework import Argument, Option
from ._output import out


DEFAULT_EXTENSION_OWNER = "ohtensorplay"
EXTENSIONS_ROOT = Path("~/.local/share/mega/extensions")
MANIFEST_FILENAME = "manifest.json"
EXTENSION_TOPIC = "mega-extension"
extensions_cli = typer_factory(
    help="Manage mega CLI extensions. Extensions execute third-party code; install only sources you trust."
)


@dataclass
class ExtensionManifest:
    owner: str
    repo: str
    repo_id: str
    short_name: str
    executable_path: str
    type: Literal["binary", "python"]
    installed_at: datetime
    description: str | None = None
    commit_sha: str | None = None

    @classmethod
    def load(cls, directory: Path) -> "ExtensionManifest":
        path = directory / MANIFEST_FILENAME
        if not path.is_file():
            raise CLIError(f"Extension manifest is missing at {path}.")
        payload = json.loads(path.read_text())
        payload = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        payload["installed_at"] = datetime.fromisoformat(payload["installed_at"])
        return cls(**payload)

    def save(self, directory: Path) -> None:
        payload = asdict(self)
        payload["installed_at"] = self.installed_at.isoformat()
        (directory / MANIFEST_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _root() -> Path:
    return EXTENSIONS_ROOT.expanduser()


def _short_name(value: str) -> str:
    name = value.rsplit("/", 1)[-1]
    if name.startswith("mega-"):
        name = name[5:]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", name):
        raise CLIError(f"Invalid extension name '{value}'. Expected [OWNER/]mega-<name>.")
    return name


def _repo(value: str) -> tuple[str, str, str]:
    if "/" in value:
        owner, repo = value.split("/", 1)
    else:
        owner, repo = DEFAULT_EXTENSION_OWNER, value
    short = _short_name(repo)
    expected = f"mega-{short}"
    if repo != expected:
        raise CLIError(f"Extension repository must be named '{expected}'.")
    return owner, repo, short


def _manifest_dir(short_name: str) -> Path:
    return _root() / short_name


def _installed() -> list[ExtensionManifest]:
    root = _root()
    if not root.is_dir():
        return []
    manifests: list[ExtensionManifest] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        try:
            manifests.append(ExtensionManifest.load(directory))
        except Exception as error:
            out.warning(f"Ignoring broken extension '{directory.name}': {error}")
    return manifests


def list_installed_extensions_for_help() -> list[tuple[str, str]]:
    return [(item.short_name, item.description or f"Extension from {item.repo_id}") for item in _installed()]


def _executable(manifest: ExtensionManifest) -> Path:
    path = Path(manifest.executable_path).expanduser()
    if not path.is_file():
        raise CLIError(f"Extension '{manifest.short_name}' is installed but its executable is missing.")
    return path


def _github_repo(owner: str, repo: str) -> dict:
    response = get_session().get(f"https://api.github.com/repos/{owner}/{repo}", timeout=15)
    mega_raise_for_status(response)
    return response.json()


def _install(repo_id: str, *, force: bool) -> ExtensionManifest:
    owner, repo, short = _repo(repo_id)
    destination = _manifest_dir(short)
    if destination.exists() and not force:
        raise CLIError(f"Extension '{short}' is already installed. Use --force to overwrite it.")
    metadata = _github_repo(owner, repo)
    branch = str(metadata.get("default_branch") or "main")
    with tempfile.TemporaryDirectory(prefix="mega-extension-") as temporary:
        checkout = Path(temporary) / repo
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, f"https://github.com/{owner}/{repo}.git", str(checkout)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise CLIError(result.stderr.strip() or f"Could not clone {owner}/{repo}.")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True, check=True).stdout.strip()
        staged = Path(temporary) / "installed"
        staged.mkdir()
        binary = checkout / repo
        if not binary.is_file():
            binary = checkout / f"mega-{short}"
        if binary.is_file():
            target = staged / f"mega-{short}"
            shutil.copy2(binary, target)
            target.chmod(target.stat().st_mode | 0o111)
            kind: Literal["binary", "python"] = "binary"
        elif (checkout / "pyproject.toml").is_file() or (checkout / "setup.py").is_file():
            environment = staged / "venv"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            result = subprocess.run([str(python), "-m", "pip", "install", str(checkout)], capture_output=True, text=True)
            if result.returncode != 0:
                raise CLIError(result.stderr.strip() or f"Could not install Python extension {owner}/{repo}.")
            target = environment / ("Scripts" if os.name == "nt" else "bin") / f"mega-{short}"
            if not target.is_file():
                raise CLIError(f"Python package does not install a 'mega-{short}' console script.")
            kind = "python"
        else:
            raise CLIError(f"{owner}/{repo} contains neither a mega-{short} executable nor a Python package.")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staged), destination)
    executable = destination / f"mega-{short}" if kind == "binary" else destination / "venv" / ("Scripts" if os.name == "nt" else "bin") / f"mega-{short}"
    manifest = ExtensionManifest(
        owner=owner,
        repo=repo,
        repo_id=f"{owner}/{repo}",
        short_name=short,
        executable_path=str(executable),
        type=kind,
        installed_at=datetime.now(timezone.utc),
        description=metadata.get("description"),
        commit_sha=commit,
    )
    manifest.save(destination)
    return manifest


@extensions_cli.command("install", examples=["mega extensions install mega-mount", "mega extensions install owner/mega-tool"])
def extension_install(
    ctx: click.Context,
    repo_id: Annotated[str, Argument(help="GitHub repository in [OWNER/]mega-<name> form.")],
    force: Annotated[bool, Option("--force", help="Overwrite an installed extension.")] = False,
) -> None:
    """Install a public GitHub extension."""
    short = _short_name(repo_id)
    reserved = set(getattr(ctx.find_root().command, "commands", {}))
    if short in reserved:
        raise CLIError(f"Extension '{short}' conflicts with an existing mega command.")
    manifest = _install(repo_id, force=force)
    out.result("Extension installed", source=manifest.repo_id, command=f"mega {manifest.short_name}", type=manifest.type)


@extensions_cli.command("update", examples=["mega extensions update", "mega extensions update mount"])
def extension_update(name: Annotated[str | None, Argument(help="Extension name; omit to update all.")] = None) -> None:
    """Reinstall extensions from their recorded GitHub sources."""
    manifests = _installed()
    if name is not None:
        short = _short_name(name)
        manifests = [item for item in manifests if item.short_name == short]
        if not manifests:
            raise CLIError(f"Extension '{short}' is not installed.")
    if not manifests:
        out.warning("No extensions installed.")
        return
    updated = []
    for manifest in manifests:
        _install(manifest.repo_id, force=True)
        updated.append(manifest.short_name)
    out.result("Extensions updated", names=updated)


def execute_extension(short_name: str, args: list[str]) -> int:
    manifest = ExtensionManifest.load(_manifest_dir(_short_name(short_name)))
    return subprocess.call([str(_executable(manifest)), *args])


@extensions_cli.command(
    "exec",
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False, "ignore_unknown_options": True},
    examples=["mega extensions exec mount -- --help"],
)
def extension_exec(ctx: click.Context, name: Annotated[str, Argument(help="Installed extension name.")]) -> None:
    """Execute an installed extension."""
    raise click.exceptions.Exit(code=execute_extension(name, list(ctx.args)))


@extensions_cli.command("list | ls", examples=["mega extensions list"])
def extension_list() -> None:
    """List installed extension commands."""
    out.table(
        [{"command": f"mega {item.short_name}", "source": item.repo_id, "type": item.type, "installed": item.installed_at.strftime("%Y-%m-%d"), "description": item.description} for item in _installed()],
        id_key="command",
    )


@extensions_cli.command("search", examples=["mega extensions search"])
def extension_search() -> None:
    """Search GitHub repositories tagged with the mega-extension topic."""
    response = get_session().get(
        "https://api.github.com/search/repositories",
        params={"q": f"topic:{EXTENSION_TOPIC}", "sort": "stars", "order": "desc", "per_page": 100},
        timeout=15,
    )
    mega_raise_for_status(response)
    installed = {item.short_name for item in _installed()}
    rows = []
    for repo in response.json().get("items", []):
        try:
            short = _short_name(repo["name"])
        except CLIError:
            continue
        rows.append({"name": short, "source": repo["full_name"], "stars": repo.get("stargazers_count", 0), "installed": short in installed, "description": repo.get("description")})
    out.table(rows, id_key="name")


@extensions_cli.command("remove | rm", examples=["mega extensions remove mount"])
def extension_remove(
    name: Annotated[str, Argument(help="Installed extension name.")],
    yes: Annotated[bool, Option("-y", "--yes", help="Skip confirmation.")] = False,
) -> None:
    """Remove an installed extension."""
    short = _short_name(name)
    directory = _manifest_dir(short)
    if not directory.is_dir():
        raise CLIError(f"Extension '{short}' is not installed.")
    out.confirm(f"Remove extension '{short}'?", yes=yes)
    shutil.rmtree(directory)
    out.result("Extension removed", name=short)


def dynamic_extension_command(name: str) -> click.Command | None:
    """Return a pass-through Click command for an installed top-level extension."""
    if not _manifest_dir(name).is_dir():
        return None

    @click.command(name=name, context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def command(args: tuple[str, ...]) -> None:
        raise click.exceptions.Exit(code=execute_extension(name, list(args)))

    command.help = next((description for short, description in list_installed_extensions_for_help() if short == name), None)
    return command


class ExtensionsAwareGroup(MegaCliGroup):
    """Root command group that exposes installed extensions as top-level commands."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        commands = set(super().list_commands(ctx))
        commands.update(name for name, _ in list_installed_extensions_for_help())
        return sorted(commands)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        return command if command is not None else dynamic_extension_command(cmd_name)

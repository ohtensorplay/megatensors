"""MEGA Hub command registry."""

import os
import sys
import traceback
from typing import Annotated

import click

from megatensors import __version__
from megatensors._hub.cli._framework import build_command as build_artifact_command
from megatensors._hub import constants
from megatensors._hub.cli._cli_utils import (
    _fetch_latest_pypi_version,
    check_cli_update,
    run_update,
    typer_factory,
)
from megatensors._hub.cli._cp import CP_EXAMPLES, make_cp
from megatensors._hub.cli._errors import format_known_exception
from megatensors._hub.cli._output import OutputFormat, out
from megatensors._hub.cli.auth import auth_cli
from megatensors._hub.cli.artifacts import convert, sign
from megatensors._hub.cli.cache import cache_cli
from megatensors._hub.cli.buckets import buckets_cli, sync
from megatensors._hub.cli.collections import collections_cli
from megatensors._hub.cli.download import DOWNLOAD_EXAMPLES, download, snapshot
from megatensors._hub.cli.discussions import discussions_cli
from megatensors._hub.cli.models import datasets_cli, models_cli, spaces_cli
from megatensors._hub.cli.papers import papers_cli
from megatensors._hub.cli.repos import repo_files_cli, repos_cli
from megatensors._hub.cli.jobs import jobs_cli
from megatensors._hub.cli.webhooks import webhooks_cli
from megatensors._hub.cli.extensions import ExtensionsAwareGroup, extensions_cli
from megatensors._hub.cli.sandbox import sandbox_cli
from megatensors._hub.cli.skills import skills_cli
from megatensors._hub.utils import dump_environment_info
from megatensors._hub.cli.upload import (
    UPLOAD_EXAMPLES,
    UPLOAD_LARGE_FOLDER_EXAMPLES,
    upload,
    upload_large_folder,
)
from megatensors._hub.utils import logging

from ._completion import _COMPLETE_VAR, InstallCompletionOpt, ShowCompletionOpt
from ._framework import Option

app = typer_factory(help="MEGA Hub CLI", cls=ExtensionsAwareGroup)
app.add_command(build_artifact_command(convert, name="convert"), "convert")
app.add_command(build_artifact_command(sign, name="sign"), "sign")
app.add_command(build_artifact_command(snapshot, name="snapshot"), "snapshot")


def env() -> None:
    """Print information about the MEGA client environment."""
    dump_environment_info()


def version() -> None:
    """Print the MEGA CLI version."""
    out.result("mega version", version=__version__)


def update() -> None:
    """Update the `mega` CLI to the latest release."""
    out.text(f"Current version: {__version__}")
    out.text("Checking for updates...")
    latest_version = _fetch_latest_pypi_version()
    if latest_version is not None and __version__ == latest_version:
        out.text(f"mega is up to date ({__version__})")
        return
    returncode = run_update()
    if returncode != 0:
        raise click.exceptions.Exit(code=returncode)


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise click.exceptions.Exit()


def _quiet_requested(args: list[str]) -> bool:
    """Detect the shared quiet format before Click resolves the leaf command."""
    return _requested_output_format(args) == OutputFormat.quiet


def _requested_output_format(args: list[str]) -> OutputFormat | None:
    """Detect shared output format flags before Click resolves the leaf command."""
    output_format_values = {format.value for format in OutputFormat}
    for index, arg in enumerate(args):
        if arg == "--":
            return None
        if arg in {"-q", "--quiet", "--format=quiet"}:
            return OutputFormat.quiet
        if arg == "--json" or arg == "--format=json":
            return OutputFormat.json
        if arg == "--format=agent":
            return OutputFormat.agent
        if arg == "--format=human":
            return OutputFormat.human
        if arg == "--format" and index + 1 < len(args):
            value = args[index + 1]
            if value in output_format_values:
                return OutputFormat(value)
    return None


@app.group_callback(invoke_without_command=True)
def app_callback(
    version: Annotated[
        bool | None,
        Option(
            "-v", "--version", callback=_version_callback, is_eager=True, hidden=True
        ),
    ] = None,
    install_completion: InstallCompletionOpt = False,
    show_completion: ShowCompletionOpt = False,
) -> None:
    pass


app.command(examples=DOWNLOAD_EXAMPLES)(download)
app.command(examples=UPLOAD_EXAMPLES)(upload)
app.command(examples=UPLOAD_LARGE_FOLDER_EXAMPLES)(upload_large_folder)
app.command(examples=CP_EXAMPLES)(make_cp())
app.command()(sync)

app.command(topic="help")(env)
app.command(topic="help")(update)
app.command(topic="help")(version)

app.add_group(auth_cli, name="auth")
app.add_group(cache_cli, name="cache")
app.add_group(buckets_cli, name="buckets")
app.add_group(collections_cli, name="collections")
app.add_group(discussions_cli, name="discussions")
app.add_group(models_cli, name="models")
app.add_group(papers_cli, name="papers")
app.add_group(datasets_cli, name="datasets")
app.add_group(spaces_cli, name="spaces")
app.add_group(repos_cli, name="repos")
app.add_group(repo_files_cli, name="repo-files")
app.add_group(jobs_cli, name="jobs")
app.add_group(webhooks_cli, name="webhooks")
app.add_group(sandbox_cli, name="sandbox")
app.add_group(extensions_cli, name="extensions | ext")
app.add_group(skills_cli, name="skills")


def main() -> None:
    if _COMPLETE_VAR not in os.environ:
        if not constants.MEGA_DEBUG:
            logging.set_verbosity_info()
        requested_format = _requested_output_format(sys.argv[1:])
        if requested_format is not None:
            out.set_mode(requested_format)
        out.banner()
        check_cli_update()

    try:
        app()
    except Exception as error:
        message = format_known_exception(error)
        if message:
            out.error(message)
            if constants.MEGA_DEBUG:
                traceback.print_exc()
            else:
                out.hint("Set MEGA_DEBUG=1 for a full traceback.")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()

# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""Contains commands to authenticate to MEGA Hub and interact with your repositories."""

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import click
from megatensors.hub import MegaHubClient

from .._login import _is_logged_in, _save_oauth_token, auth_list, auth_switch, login, logout
from ..errors import CLIError
from ..utils import get_stored_tokens, get_token, logging, select_choice
from ..utils._oauth_device import poll_device_token, request_device_code
from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import OutputFormat, _MEGA_ASCII, out

logger = logging.get_logger(__name__)


def print_login_header() -> None:
    out.banner()


auth_cli = typer_factory(help="Manage authentication (login, logout, etc.).")
keys_cli = typer_factory(help="Manage SSH and GPG public keys for your MEGA account.")
auth_cli.add_group(keys_cli, name="keys")


class AccountKeyType(str, Enum):
    ssh = "ssh"
    gpg = "gpg"


@auth_cli.command(
    "login",
    examples=[
        "mega auth login",
        "mega auth login --token $MEGA_TOKEN",
        "mega auth login --token $MEGA_TOKEN --add-to-git-credential",
        "mega auth login --force",
    ],
)
def auth_login(
    token: TokenOpt = None,
    add_to_git_credential: Annotated[
        bool,
        Option(
            help="Save to git credential helper. Useful only if you plan to run git commands directly.",
        ),
    ] = False,
    force: Annotated[
        bool,
        Option(
            help="Force re-login even if already logged in.",
        ),
    ] = False,
) -> None:
    """Log in from your browser, or using a MEGA access token."""
    if out.mode == OutputFormat.human:
        print_login_header()
    if token is not None or out.mode == OutputFormat.human:
        # `--token` bypasses any prompt; in human mode the gh-style menu lives in `login()`.
        login(
            token=token,
            add_to_git_credential=add_to_git_credential,
            skip_if_logged_in=not force,
        )
        return

    # Logging in is an interactive flow: besides human mode, only agent mode is supported.
    if out.mode != OutputFormat.agent:
        raise CLIError(
            "`mega auth login` is interactive and does not support --format json/quiet. "
            "Pass --token for a non-interactive login."
        )

    # agent mode: never prompt; print instructions the agent can relay to its user.
    if not force and _is_logged_in():
        out.text(agent="Already logged in. Use `mega auth login --force` to re-login.")
        return
    device_info = request_device_code()
    out.text(
        agent=(
            f"Ask the user to open {device_info['verification_uri_complete']} in a browser and enter the code "
            f"{device_info['user_code']}. The code expires in {device_info['expires_in']} seconds. "
            "Waiting for authorization..."
        )
    )
    response = poll_device_token(device_info)
    _, username = _save_oauth_token(response)
    out.text(agent=f"Login successful: logged in as {username}.")


@auth_cli.command(
    "logout",
    examples=["mega auth logout", "mega auth logout --token-name my-token"],
)
def auth_logout(
    token_name: Annotated[
        str | None,
        Option(help="Name of token to logout"),
    ] = None,
) -> None:
    """Logout from a specific token."""
    logout(token_name=token_name)


def _select_token_name() -> str | None:
    token_names = list(get_stored_tokens().keys())

    if not token_names:
        logger.error("No stored tokens found. Please login first.")
        return None

    if out.mode != OutputFormat.human:
        raise CLIError("Use --token-name to select a token in non-interactive mode.")
    return token_names[select_choice("Select a token to switch to:", token_names)]


@auth_cli.command(
    "switch",
    examples=["mega auth switch", "mega auth switch --token-name my-token"],
)
def auth_switch_cmd(
    token_name: Annotated[
        str | None,
        Option(
            help="Name of the token to switch to",
        ),
    ] = None,
    add_to_git_credential: Annotated[
        bool,
        Option(
            help="Save to git credential helper. Useful only if you plan to run git commands directly.",
        ),
    ] = False,
) -> None:
    """Switch between access tokens."""
    if token_name is None:
        token_name = _select_token_name()
    if token_name is None:
        out.error(
            "No token name provided. Run `mega auth login` first or pass `--token-name`."
        )
        raise click.exceptions.Exit(code=1)
    auth_switch(token_name, add_to_git_credential=add_to_git_credential)


@auth_cli.command("list | ls", examples=["mega auth list"])
def auth_list_cmd() -> None:
    """List all stored access tokens."""
    auth_list()


@auth_cli.command(
    "token",
    examples=[
        "mega auth token",
        "mega auth token | xargs curl -H 'Authorization: Bearer {}'",
    ],
)
def auth_token() -> None:
    """Print the current access token to stdout."""
    token = get_token()
    if token is None:
        out.error("Not logged in. Run `mega auth login` first.")
        raise click.exceptions.Exit(code=1)
    print(token)
    out.hint("Run `mega auth whoami` to see which account this token belongs to.")


@auth_cli.command(
    "whoami", examples=["mega auth whoami", "mega auth whoami --format json"]
)
def auth_whoami() -> None:
    """Find out which MEGA account you are logged in as."""

    token = get_token()
    if token is None:
        out.error("Not logged in")
        raise click.exceptions.Exit(code=1)

    info = MegaHubClient(token=token).whoami()
    access_token = (info.get("auth") or {}).get("accessToken") or {}
    orgs = (
        ",".join(
            str(org.get("name")) for org in info.get("orgs", []) if org.get("name")
        )
        or None
    )
    scopes = access_token.get("scopes")
    out.result(
        "Logged in",
        user=info["name"],
        orgs=orgs,
        token_kind=access_token.get("kind"),
        role=access_token.get("role"),
        scopes=",".join(scopes) if isinstance(scopes, list) else None,
    )


@keys_cli.command(
    "list | ls",
    examples=["mega auth keys list", "mega auth keys list --format json"],
)
def account_keys_list(token: TokenOpt = None) -> None:
    """List active public keys registered to your MEGA account."""
    keys = MegaHubClient(token=token).list_account_keys()
    out.table(
        [
            {
                "id": key.key_id,
                "type": key.key_type,
                "name": key.name,
                "fingerprint": key.fingerprint,
                "created_at": key.created_at,
            }
            for key in keys
        ],
        id_key="id",
    )


@keys_cli.command(
    "add",
    examples=[
        "mega auth keys add ~/.ssh/id_ed25519.pub --name 'Work laptop'",
        "mega auth keys add signing-key.asc --type gpg --name release",
    ],
)
def account_keys_add(
    public_key_file: Annotated[
        Path,
        Argument(
            help="SSH .pub file, armored GPG public-key file, or '-' to read a public key from stdin."
        ),
    ],
    name: Annotated[
        str | None, Option("--name", help="A recognizable name for this key.")
    ] = None,
    key_type: Annotated[
        AccountKeyType | None,
        Option("--type", help="Public-key type. Auto-detected when omitted."),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Add an SSH or GPG public key to your MEGA account."""
    public_key = _read_public_key(public_key_file)
    resolved_type = key_type or _detect_key_type(public_key)
    resolved_name = (
        name or ("stdin key" if str(public_key_file) == "-" else public_key_file.name)
    ).strip()
    if not resolved_name:
        raise CLIError("Key name cannot be empty.")
    key = MegaHubClient(token=token).add_account_key(
        key_type=resolved_type.value,
        name=resolved_name,
        public_key=public_key,
    )
    out.result(
        "Public key added",
        key_id=key.key_id,
        type=key.key_type,
        name=key.name,
        fingerprint=key.fingerprint,
    )


@keys_cli.command(
    "delete | remove | rm",
    examples=["mega auth keys delete <key-id>", "mega auth keys rm <key-id> --yes"],
)
def account_keys_delete(
    key_id: Annotated[str, Argument(help="Public key ID to remove.")],
    yes: Annotated[
        bool, Option("--yes", "-y", help="Skip the confirmation prompt.")
    ] = False,
    token: TokenOpt = None,
) -> None:
    """Remove an active public key from your MEGA account."""
    out.confirm(f"Remove public key '{key_id}'?", yes=yes)
    MegaHubClient(token=token).delete_account_key(key_id)
    out.result("Public key removed", key_id=key_id)


def _read_public_key(source: Path) -> str:
    stream = None
    try:
        stream = sys.stdin if str(source) == "-" else source.open("r", encoding="utf-8")
        first_line = stream.readline(64_001)
        if "PRIVATE KEY" in first_line.upper():
            raise CLIError(
                "Private keys are never accepted. Provide only an SSH .pub file or an armored GPG public key."
            )
        if not first_line.lstrip().startswith(
            ("ssh-", "ecdsa-", "sk-", "-----BEGIN PGP PUBLIC KEY BLOCK-----")
        ):
            raise CLIError(
                "Input does not start with a supported SSH or GPG public-key header."
            )
        value = first_line + stream.read(64_001)
    except (OSError, UnicodeError) as error:
        raise CLIError(f"Cannot read public key from '{source}': {error}") from error
    finally:
        if stream is not None and stream is not sys.stdin:
            stream.close()
    if not value.strip():
        raise CLIError("Public key input is empty.")
    if len(value.encode("utf-8")) > 64_000:
        raise CLIError("Public key input exceeds 64 KB.")
    if "PRIVATE KEY" in value.upper():
        raise CLIError(
            "Private keys are never accepted. Provide only an SSH .pub file or an armored GPG public key."
        )
    return value


def _detect_key_type(public_key: str) -> AccountKeyType:
    stripped = public_key.lstrip()
    if stripped.startswith("-----BEGIN PGP PUBLIC KEY BLOCK-----"):
        return AccountKeyType.gpg
    if stripped.startswith(("ssh-", "ecdsa-", "sk-")):
        return AccountKeyType.ssh
    raise CLIError("Cannot detect public-key type. Pass --type ssh or --type gpg.")

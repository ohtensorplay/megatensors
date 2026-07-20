# SPDX-License-Identifier: Apache-2.0
"""Production webhook-route commands for the MEGA Hub CLI.

The command layout intentionally follows the upstream Hub CLI convention:
``list``, ``info``, ``create``, ``update``, ``enable``, ``disable``, and
``delete`` all map one-to-one to a stable service API. MEGA additionally
exposes ``test`` and ``deliveries`` because delivery receipts are part of the
operational contract, not a browser-only diagnostic.
"""

from functools import wraps
from typing import Annotated, Any, Callable, TypeVar

from megatensors.hub import MegaHubClient, MegaHubError, WebhookDeliveryInfo, WebhookInfo, WebhookLastDeliveryInfo

from megatensors._hub.errors import CLIError
from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out


webhooks_cli = typer_factory(help="Manage signed MEGA webhook routes and delivery receipts.")

_WEBHOOK_EVENTS = {
    "repo.created",
    "repo.updated",
    "repo.deleted",
    "repo.ref.created",
    "repo.ref.deleted",
    "discussion.created",
    "discussion.updated",
    "discussion.reply.created",
}

F = TypeVar("F", bound=Callable[..., Any])


def _worker_errors_as_cli_errors(command: F) -> F:
    """Present MEGA API failures through the shared CLI error formatter."""

    @wraps(command)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except MegaHubError as error:
            raise CLIError(str(error)) from error

    return wrapped  # type: ignore[return-value]


def _validated_events(events: list[str] | None) -> list[str] | None:
    if events is None:
        return None
    selected = [event.strip() for event in events if event.strip()]
    if not selected:
        raise CLIError("Provide at least one --event.")
    invalid = sorted(set(selected) - _WEBHOOK_EVENTS)
    if invalid:
        choices = ", ".join(sorted(_WEBHOOK_EVENTS))
        raise CLIError(f"Unsupported webhook event: {', '.join(invalid)}. Choose from: {choices}.")
    return list(dict.fromkeys(selected))


def _webhook_record(webhook: WebhookInfo) -> dict[str, Any]:
    return {
        "webhook_id": webhook.webhook_id,
        "name": webhook.name,
        "url": webhook.url,
        "scope": webhook.scope,
        "repo_id": webhook.repo_id,
        "events": list(webhook.events),
        "enabled": webhook.enabled,
        "secret_configured": webhook.secret_configured,
        "created_at": webhook.created_at,
        "updated_at": webhook.updated_at,
        "last_delivery": _last_delivery_record(webhook.last_delivery) if webhook.last_delivery else None,
    }


def _delivery_record(delivery: WebhookDeliveryInfo | None) -> dict[str, Any] | None:
    if delivery is None:
        return None
    return {
        "delivery_id": delivery.delivery_id,
        "webhook_id": delivery.webhook_id,
        "event": delivery.event,
        "state": delivery.state,
        "attempt_count": delivery.attempt_count,
        "response_status": delivery.response_status,
        "error": delivery.error,
        "created_at": delivery.created_at,
        "delivered_at": delivery.delivered_at,
    }


def _last_delivery_record(delivery: WebhookLastDeliveryInfo) -> dict[str, Any]:
    return {
        "state": delivery.state,
        "response_status": delivery.response_status,
        "error": delivery.error,
        "delivered_at": delivery.delivered_at,
    }


@webhooks_cli.command(
    "list | ls",
    examples=["mega webhooks list", "mega webhooks ls --format json", "mega webhooks ls --format quiet"],
)
@_worker_errors_as_cli_errors
def webhooks_list(token: TokenOpt = None) -> None:
    """List signed webhook routes owned by the current account."""
    webhooks = MegaHubClient(token=token).list_webhooks()
    out.table(
        [
            {
                "id": webhook.webhook_id,
                "name": webhook.name,
                "scope": webhook.scope,
                "target": webhook.url,
                "enabled": webhook.enabled,
                "events": ",".join(webhook.events),
                "last_delivery": webhook.last_delivery.state if webhook.last_delivery else None,
            }
            for webhook in webhooks
        ],
        id_key="id",
    )


@webhooks_cli.command("info", examples=["mega webhooks info <webhook-id>", "mega webhooks info <webhook-id> --format json"])
@_worker_errors_as_cli_errors
def webhooks_info(
    webhook_id: Annotated[str, Argument(help="Webhook route ID.")],
    token: TokenOpt = None,
) -> None:
    """Show complete configuration and last-delivery state for one route."""
    out.dict(_webhook_record(MegaHubClient(token=token).get_webhook(webhook_id)), id_key="webhook_id")


@webhooks_cli.command(
    "create",
    examples=[
        "mega webhooks create --name release-verifier --url https://ci.example/hooks/mega --event repo.updated",
        "mega webhooks create --name community-index --url https://index.example/mega --repo mega/catalog --event discussion.created --event discussion.reply.created",
    ],
)
@_worker_errors_as_cli_errors
def webhooks_create(
    name: Annotated[str, Option("--name", help="A recognizable label for this route.")],
    url: Annotated[str, Option("--url", help="Public HTTPS receiver URL.")],
    event: Annotated[list[str], Option("--event", help="Event to relay. Repeatable.")],
    repo_id: Annotated[str | None, Option("--repo", help="Limit the route to one owner/repository.")] = None,
    secret: Annotated[str | None, Option("--secret", help="Signing secret. A one-time secret is generated when omitted.")] = None,
    token: TokenOpt = None,
) -> None:
    """Create a signed account or single-repository webhook route."""
    events = _validated_events(event)
    assert events is not None
    route = MegaHubClient(token=token).create_webhook(
        name=name,
        url=url,
        events=events,
        scope="repository" if repo_id else "account",
        repo_id=repo_id,
        secret=secret,
    )
    out.result(
        "Webhook created",
        webhook_id=route.webhook.webhook_id,
        signing_secret=route.signing_secret,
    )
    out.hint("Store the signing secret now; MEGA does not reveal it again. Use `mega webhooks test <id>` to enqueue a signed test.")


@webhooks_cli.command(
    "update",
    examples=[
        "mega webhooks update <webhook-id> --url https://ci.example/new-hook",
        "mega webhooks update <webhook-id> --event repo.created --event repo.updated",
        "mega webhooks update <webhook-id> --secret rotate-this-secret",
    ],
)
@_worker_errors_as_cli_errors
def webhooks_update(
    webhook_id: Annotated[str, Argument(help="Webhook route ID.")],
    name: Annotated[str | None, Option("--name", help="Replace the route label.")] = None,
    url: Annotated[str | None, Option("--url", help="Replace the public HTTPS receiver URL.")] = None,
    event: Annotated[list[str] | None, Option("--event", help="Replace the event set. Repeatable.")] = None,
    secret: Annotated[str | None, Option("--secret", help="Rotate the signing secret.")] = None,
    token: TokenOpt = None,
) -> None:
    """Update provided fields of one route; omitted fields remain unchanged."""
    events = _validated_events(event)
    if name is None and url is None and events is None and secret is None:
        raise CLIError("Provide at least one field to update.")
    webhook = MegaHubClient(token=token).update_webhook(
        webhook_id,
        name=name,
        url=url,
        events=events,
        secret=secret,
    )
    out.result("Webhook updated", webhook_id=webhook.webhook_id, enabled=webhook.enabled)


@webhooks_cli.command("enable", examples=["mega webhooks enable <webhook-id>"])
@_worker_errors_as_cli_errors
def webhooks_enable(
    webhook_id: Annotated[str, Argument(help="Webhook route ID.")],
    token: TokenOpt = None,
) -> None:
    """Enable a paused route."""
    webhook = MegaHubClient(token=token).update_webhook(webhook_id, enabled=True)
    out.result("Webhook enabled", webhook_id=webhook.webhook_id)


@webhooks_cli.command("disable", examples=["mega webhooks disable <webhook-id>"])
@_worker_errors_as_cli_errors
def webhooks_disable(
    webhook_id: Annotated[str, Argument(help="Webhook route ID.")],
    token: TokenOpt = None,
) -> None:
    """Pause a route and cancel any unstarted deliveries."""
    webhook = MegaHubClient(token=token).update_webhook(webhook_id, enabled=False)
    out.result("Webhook disabled", webhook_id=webhook.webhook_id)


@webhooks_cli.command("test", examples=["mega webhooks test <webhook-id>", "mega webhooks test <webhook-id> --format json"])
@_worker_errors_as_cli_errors
def webhooks_test(
    webhook_id: Annotated[str, Argument(help="Enabled webhook route ID.")],
    token: TokenOpt = None,
) -> None:
    """Enqueue one signed test payload without waiting for a receiver response."""
    delivery = MegaHubClient(token=token).test_webhook(webhook_id)
    out.dict(_delivery_record(delivery), id_key="delivery_id")


@webhooks_cli.command(
    "deliveries | delivery",
    examples=["mega webhooks deliveries <webhook-id>", "mega webhooks deliveries <webhook-id> --limit 50 --format json"],
)
@_worker_errors_as_cli_errors
def webhook_deliveries(
    webhook_id: Annotated[str, Argument(help="Webhook route ID.")],
    limit: Annotated[int, Option("--limit", help="Maximum delivery receipts to return (1-100).", min=1)] = 20,
    token: TokenOpt = None,
) -> None:
    """Inspect retained delivery receipts, queue state, and receiver outcomes."""
    deliveries = MegaHubClient(token=token).list_webhook_deliveries(webhook_id, limit=limit)
    out.table(
        [
            {
                "id": delivery.delivery_id,
                "event": delivery.event,
                "state": delivery.state,
                "attempts": delivery.attempt_count,
                "status": delivery.response_status,
                "created_at": delivery.created_at,
                "delivered_at": delivery.delivered_at,
                "error": delivery.error,
            }
            for delivery in deliveries
        ],
        id_key="id",
    )


@webhooks_cli.command(
    "delete | remove | rm",
    examples=["mega webhooks delete <webhook-id>", "mega webhooks rm <webhook-id> --yes"],
)
@_worker_errors_as_cli_errors
def webhooks_delete(
    webhook_id: Annotated[str, Argument(help="Webhook route ID.")],
    yes: Annotated[bool, Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a route and its retained delivery receipts."""
    out.confirm(f"Delete webhook '{webhook_id}' and its delivery history?", yes=yes)
    MegaHubClient(token=token).delete_webhook(webhook_id)
    out.result("Webhook deleted", webhook_id=webhook_id)

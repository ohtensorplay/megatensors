"""Commands for managing MEGA Hub collections."""

import enum
from typing import Annotated

from megatensors.hub import MegaHubClient

from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import out


class CollectionItemType(str, enum.Enum):
    model = "model"
    dataset = "dataset"
    space = "space"
    paper = "paper"
    collection = "collection"
    bucket = "bucket"


class CollectionSort(str, enum.Enum):
    lastModified = "lastModified"
    trending = "trending"
    upvotes = "upvotes"


collections_cli = typer_factory(help="Manage collections on the MEGA Hub.")


def _client(token: str | None) -> MegaHubClient:
    return MegaHubClient(token=token)


@collections_cli.command(
    "list | ls",
    examples=[
        "mega collections ls",
        "mega collections ls --owner mega",
        "mega collections ls --item models/mega/mega-3b-base --limit 10",
    ],
)
def collections_ls(
    owner: Annotated[str | None, Option(help="Filter by owner username or organization.")] = None,
    item: Annotated[
        str | None,
        Option(help='Filter by an item such as "models/mega/mega-3b-base" or "papers/2503.00948".'),
    ] = None,
    sort: Annotated[
        CollectionSort | None,
        Option(help="Sort by last modified, trending, or upvotes."),
    ] = None,
    search: Annotated[str | None, Option("--search", "-s", help="Filter collection title or description.")] = None,
    limit: Annotated[int, Option(help="Maximum collections to return.", min=1)] = 20,
    token: TokenOpt = None,
) -> None:
    """List collections visible to the current identity."""
    collections = _client(token).list_collections(
        owner=owner,
        item=item,
        sort=sort.value if sort else None,
        search=search,
        limit=limit,
    )
    out.table(
        [
            {
                "slug": collection.slug,
                "title": collection.title,
                "items": collection.item_count,
                "upvotes": collection.upvotes,
                "visibility": "private" if collection.private else "public",
                "theme": collection.theme,
                "updated_at": collection.updated_at,
            }
            for collection in collections
        ],
        id_key="slug",
    )


@collections_cli.command("info", examples=["mega collections info username/my-collection"])
def collections_info(
    collection_slug: Annotated[str, Argument(help="Collection slug in owner/slug form.")],
    token: TokenOpt = None,
) -> None:
    """Show collection metadata and ordered items."""
    out.dict(_client(token).get_collection(collection_slug), id_key="slug")


@collections_cli.command(
    "create",
    examples=[
        'mega collections create "My Models"',
        'mega collections create "My Models" --description "Favorite releases" --private',
        'mega collections create "Org Collection" --namespace my-org',
    ],
)
def collections_create(
    title: Annotated[str, Argument(help="Collection title.")],
    namespace: Annotated[str | None, Option(help="Username or organization; defaults to the current user.")] = None,
    description: Annotated[str | None, Option(help="Collection description.")] = None,
    private: Annotated[bool, Option(help="Create a private collection.")] = False,
    theme: Annotated[str | None, Option(help="Theme color such as green, blue, or orange.")] = None,
    exists_ok: Annotated[bool, Option(help="Return an existing collection with the same generated slug.")] = False,
    token: TokenOpt = None,
) -> None:
    """Create a collection."""
    collection = _client(token).create_collection(
        title,
        namespace=namespace,
        description=description,
        private=private,
        theme=theme,
        exists_ok=exists_ok,
    )
    out.result("Collection created", slug=collection.slug, url=collection.url)


@collections_cli.command(
    "update",
    examples=[
        'mega collections update username/my-collection --title "New title"',
        'mega collections update username/my-collection --description "Updated description"',
        "mega collections update username/my-collection --private --theme green",
    ],
)
def collections_update(
    collection_slug: Annotated[str, Argument(help="Collection slug in owner/slug form.")],
    title: Annotated[str | None, Option(help="New title.")] = None,
    description: Annotated[str | None, Option(help="New description.")] = None,
    position: Annotated[int | None, Option(help="New position in the owner's collection list.", min=0)] = None,
    private: Annotated[bool | None, Option("--private/--public", help="Set collection visibility.")] = None,
    theme: Annotated[str | None, Option(help="New theme color.")] = None,
    token: TokenOpt = None,
) -> None:
    """Update collection metadata."""
    collection = _client(token).update_collection(
        collection_slug,
        title=title,
        description=description,
        position=position,
        private=private,
        theme=theme,
    )
    out.result("Collection updated", slug=collection.slug, url=collection.url)


@collections_cli.command(
    "delete",
    examples=["mega collections delete username/my-collection", "mega collections delete username/my-collection --missing-ok"],
)
def collections_delete(
    collection_slug: Annotated[str, Argument(help="Collection slug in owner/slug form.")],
    missing_ok: Annotated[bool, Option(help="Do not fail if the collection is already absent.")] = False,
    token: TokenOpt = None,
) -> None:
    """Delete a collection."""
    _client(token).delete_collection(collection_slug, missing_ok=missing_ok)
    out.result("Collection deleted", slug=collection_slug)


@collections_cli.command(
    "add-item",
    examples=[
        "mega collections add-item username/my-collection mega/mega-3b-base model",
        'mega collections add-item username/my-collection 2503.00948 paper --note "Technical report"',
        "mega collections add-item username/my-collection mega/mega-playground space",
        'mega collections add-item username/my-collection mega/release-artifacts bucket --note "Mutable artifacts"',
    ],
)
def collections_add_item(
    collection_slug: Annotated[str, Argument(help="Collection slug in owner/slug form.")],
    item_id: Annotated[str, Argument(help="Repository, Bucket, paper, or collection ID.")],
    item_type: Annotated[CollectionItemType, Argument(help="Item type.")],
    note: Annotated[str | None, Option(help="Note attached to the item (max 500 characters).")] = None,
    position: Annotated[int | None, Option(help="Ordered position in the collection.", min=0)] = None,
    exists_ok: Annotated[bool, Option(help="Do not fail if the item is already present.")] = False,
    token: TokenOpt = None,
) -> None:
    """Add an item to a collection."""
    collection = _client(token).add_collection_item(
        collection_slug,
        item_id,
        item_type.value,
        note=note,
        position=position,
        exists_ok=exists_ok,
    )
    out.result("Item added to collection", slug=collection.slug, url=collection.url)


@collections_cli.command(
    "update-item",
    examples=[
        'mega collections update-item username/my-collection ITEM_OBJECT_ID --note "Updated note"',
        "mega collections update-item username/my-collection ITEM_OBJECT_ID --position 0",
    ],
)
def collections_update_item(
    collection_slug: Annotated[str, Argument(help="Collection slug in owner/slug form.")],
    item_object_id: Annotated[str, Argument(help="Stable item_object_id returned by collections info.")],
    note: Annotated[str | None, Option(help="New note.")] = None,
    position: Annotated[int | None, Option(help="New ordered position.", min=0)] = None,
    token: TokenOpt = None,
) -> None:
    """Update an item's note or position."""
    collection = _client(token).update_collection_item(
        collection_slug, item_object_id, note=note, position=position
    )
    out.result("Item updated in collection", slug=collection.slug)


@collections_cli.command("delete-item")
def collections_delete_item(
    collection_slug: Annotated[str, Argument(help="Collection slug in owner/slug form.")],
    item_object_id: Annotated[str, Argument(help="Stable item_object_id returned by collections info.")],
    missing_ok: Annotated[bool, Option(help="Do not fail if the item is already absent.")] = False,
    token: TokenOpt = None,
) -> None:
    """Remove an item from a collection."""
    collection = _client(token).delete_collection_item(
        collection_slug, item_object_id, missing_ok=missing_ok
    )
    out.result("Item deleted from collection", slug=collection.slug)

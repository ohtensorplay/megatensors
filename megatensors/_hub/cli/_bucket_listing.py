# SPDX-License-Identifier: Apache-2.0
"""Terminal rendering for MEGA Storage Bucket listings."""

import json
from datetime import datetime
from typing import Sequence

import click

from megatensors._hub._buckets import BucketFile, BucketFolder

from ._output import OutputFormat, _dataclass_to_dict, out


BucketItem = BucketFile | BucketFolder


def print_bucket_listing(
    items: Sequence[BucketItem],
    *,
    human_readable: bool = False,
    as_tree: bool = False,
    recursive: bool = False,
) -> None:
    """Print the same flat/tree representations exposed by ``hf buckets list``."""
    if as_tree and out.mode == OutputFormat.json:
        raise click.BadParameter("Cannot use --tree with --format json.")
    if not items:
        if out.mode == OutputFormat.json:
            print("[]")
        elif out.mode != OutputFormat.quiet:
            print("(empty)")
        return

    has_directories = any(isinstance(item, BucketFolder) for item in items)
    if as_tree:
        for line in _build_tree(items, human_readable, out.mode == OutputFormat.quiet):
            print(line)
    elif out.mode == OutputFormat.json:
        print(json.dumps([_dataclass_to_dict(item) for item in items], indent=2))
    elif out.mode == OutputFormat.quiet:
        for item in items:
            print(f"{item.path}/" if isinstance(item, BucketFolder) else item.path)
    else:
        for item in items:
            date = _format_date(_item_date(item), human_readable)
            if isinstance(item, BucketFolder):
                print(f"{'':>12}  {date:>19}  {item.path}/")
            else:
                print(f"{_format_size(item.size, human_readable):>12}  {date:>19}  {item.path}")
    if not recursive and has_directories:
        out.hint("Use -R to list files recursively.")


def _build_tree(items: Sequence[BucketItem], human_readable: bool, quiet: bool) -> list[str]:
    tree: dict[str, dict] = {}
    for item in items:
        current = tree
        parts = item.path.split("/")
        for part in parts[:-1]:
            current = current.setdefault(part, {"children": {}})["children"]
        leaf = parts[-1]
        if isinstance(item, BucketFolder):
            current.setdefault(leaf, {"children": {}})
        else:
            current[leaf] = {"item": item}

    file_prefixes = [
        (_format_size(item.size, human_readable), _format_date(_item_date(item), human_readable))
        for item in items
        if isinstance(item, BucketFile)
    ]
    size_width = max((len(size) for size, _ in file_prefixes), default=0)
    date_width = max((len(date) for _, date in file_prefixes), default=0)
    prefix_width = 0 if quiet else size_width + (2 if size_width else 0) + date_width
    lines: list[str] = []
    _render_tree(tree, lines, "", prefix_width, size_width, human_readable)
    return lines


def _render_tree(
    node: dict[str, dict],
    lines: list[str],
    indent: str,
    prefix_width: int,
    size_width: int,
    human_readable: bool,
) -> None:
    entries = sorted(node.items())
    for index, (name, value) in enumerate(entries):
        last = index == len(entries) - 1
        directory = "children" in value
        connector = "└── " if last else "├── "
        if prefix_width:
            if directory:
                prefix = " " * prefix_width
            else:
                item = value["item"]
                prefix = f"{_format_size(item.size, human_readable):>{size_width}}  {_format_date(_item_date(item), human_readable)}"
            lines.append(f"{prefix}  {indent}{connector}{name}{'/' if directory else ''}")
        else:
            lines.append(f"{indent}{connector}{name}{'/' if directory else ''}")
        if directory and value["children"]:
            _render_tree(
                value["children"],
                lines,
                indent + ("    " if last else "│   "),
                prefix_width,
                size_width,
                human_readable,
            )


def _item_date(item: BucketItem) -> datetime | None:
    if isinstance(item, BucketFile) and item.mtime is not None:
        return item.mtime
    return item.uploaded_at


def _format_size(size: int | float, human_readable: bool) -> str:
    if not human_readable:
        return str(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000:
            return f"{size:g} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} PB"


def _format_date(value: datetime | None, human_readable: bool) -> str:
    if value is None:
        return ""
    return value.strftime("%b %d %H:%M" if human_readable else "%Y-%m-%d %H:%M:%S")

"""Commands for discovering and reading source-grounded MEGA paper records."""

import datetime
import enum
from typing import Annotated

import click

from megatensors import MegaApi
from megatensors._hub.errors import CLIError, MegaHubHTTPError

from ._cli_utils import TokenOpt, typer_factory
from ._framework import Argument, Option
from ._output import OutputFormat, _dataclass_to_dict, out


class PaperSort(str, enum.Enum):
    publishedAt = "publishedAt"
    trending = "trending"


def _parse_date(value: str | None) -> str | None:
    if value is None:
        return None
    return datetime.date.today().isoformat() if value.lower() == "today" else value


papers_cli = typer_factory(help="Discover and read papers on the MEGA Hub.")


@papers_cli.command(
    "list | ls",
    examples=[
        "mega papers ls",
        "mega papers ls --sort trending",
        "mega papers ls --date 2026-07-17",
        "mega papers ls --week 2026-W29",
        "mega papers ls --month 2026-07",
    ],
)
def papers_ls(
    date: Annotated[str | None, Option(help="Date in YYYY-MM-DD format or 'today'.", callback=_parse_date)] = None,
    week: Annotated[str | None, Option(help="ISO week such as 2026-W29.")] = None,
    month: Annotated[str | None, Option(help="Month in YYYY-MM format.")] = None,
    submitter: Annotated[str | None, Option(help="MEGA submitter filter in origin:handle form.")] = None,
    sort: Annotated[PaperSort | None, Option(help="Sort by publication time or trending signal.")] = None,
    limit: Annotated[int, Option(help="Maximum papers to return (up to 100).", min=1)] = 50,
    token: TokenOpt = None,
) -> None:
    """List a daily, weekly, or monthly MEGA paper edition."""
    if limit > 100:
        raise click.BadParameter("must be at most 100", param_hint="--limit")
    papers = MegaApi(token=token).list_daily_papers(
        date=date,
        week=week,
        month=month,
        submitter=submitter,
        sort=sort.value if sort else None,
        limit=limit,
    )
    results = []
    for paper in papers:
        item = _dataclass_to_dict(paper)
        submitted_by = item.get("submitted_by") or {}
        item["submitted_by_name"] = submitted_by.get("fullname") or submitted_by.get("username") or ""
        results.append(item)
    out.table(results, headers=["id", "title", "upvotes", "comments", "published_at", "submitted_by_name"])


@papers_cli.command(
    "search",
    examples=['mega papers search "vision language"', 'mega papers search "diffusion" --limit 10'],
)
def papers_search(
    query: Annotated[str, Argument(help="Search query for the official arXiv catalog.")],
    limit: Annotated[int, Option(help="Maximum papers to return (up to 50).", min=1)] = 20,
    token: TokenOpt = None,
) -> None:
    """Search the official arXiv catalog through MEGA."""
    if limit > 50:
        raise click.BadParameter("must be at most 50", param_hint="--limit")
    results = [
        _dataclass_to_dict(paper)
        for paper in MegaApi(token=token).list_papers(query=query, limit=limit)
    ]
    out.table(results, headers=["id", "title", "summary", "upvotes", "published_at"])


@papers_cli.command("info", examples=["mega papers info 2607.05394"])
def papers_info(
    paper_id: Annotated[str, Argument(help="Official arXiv paper ID.")],
    token: TokenOpt = None,
) -> None:
    """Show a source-grounded MEGA paper record and linked resources."""
    try:
        info = MegaApi(token=token).paper_info(id=paper_id)
    except MegaHubHTTPError as error:
        if error.response.status_code == 404:
            raise CLIError(f"Paper '{paper_id}' was not found on MEGA.") from error
        raise
    out.dict(info, id_key="id")


@papers_cli.command("read", examples=["mega papers read 2607.05394"])
def papers_read(
    paper_id: Annotated[str, Argument(help="Official arXiv paper ID.")],
    token: TokenOpt = None,
) -> None:
    """Read a paper as source-grounded Markdown."""
    try:
        markdown = MegaApi(token=token).read_paper(id=paper_id)
    except MegaHubHTTPError as error:
        if error.response.status_code == 404:
            raise CLIError(f"Paper '{paper_id}' was not found on MEGA.") from error
        raise
    if out.mode == OutputFormat.json:
        out.dict({"arxiv_id": paper_id, "markdown": markdown})
        return
    click.echo(markdown, nl=not markdown.endswith("\n"))

import json

from click.testing import CliRunner

from megatensors._hub.cli import mega
from megatensors._hub.cli import papers as papers_module
from megatensors._hub.mega_api import PaperInfo


def _paper() -> PaperInfo:
    return PaperInfo(
        id="2607.05394",
        title="A source-grounded paper",
        source="arxiv",
        numComments=2,
        submittedOnDailyAt="2026-07-17T08:00:00Z",
        submittedOnDailyBy={"user": "mega", "fullname": "MEGA Official"},
        paper={
            "id": "2607.05394",
            "authors": [{"name": "Ada Researcher"}],
            "publishedAt": "2026-07-16T08:00:00Z",
            "summary": "Official arXiv metadata.",
            "upvotes": 12,
        },
    )


class _FakeApi:
    calls: list[tuple[str, dict]] = []

    def __init__(self, **kwargs):
        self.calls.append(("init", kwargs))

    def list_daily_papers(self, **kwargs):
        self.calls.append(("list", kwargs))
        return iter([_paper()])

    def list_papers(self, **kwargs):
        self.calls.append(("search", kwargs))
        return iter([_paper()])

    def paper_info(self, **kwargs):
        self.calls.append(("info", kwargs))
        return _paper()

    def read_paper(self, **kwargs):
        self.calls.append(("read", kwargs))
        return f"# Test paper\n\n`arXiv:{kwargs['id']}`\n"


def _install_fake(monkeypatch):
    _FakeApi.calls = []
    monkeypatch.setattr(papers_module, "MegaApi", _FakeApi)


def test_papers_list_supports_daily_period_filters(monkeypatch):
    _install_fake(monkeypatch)

    result = CliRunner().invoke(
        mega.app,
        [
            "papers",
            "ls",
            "--date",
            "2026-07-17",
            "--sort",
            "trending",
            "--limit",
            "10",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["id"] == "2607.05394"
    assert payload[0]["submitted_by_name"] == "MEGA Official"
    assert ("list", {
        "date": "2026-07-17",
        "week": None,
        "month": None,
        "submitter": None,
        "sort": "trending",
        "limit": 10,
    }) in _FakeApi.calls


def test_papers_search_uses_the_native_catalog(monkeypatch):
    _install_fake(monkeypatch)

    result = CliRunner().invoke(
        mega.app,
        ["papers", "search", "vision language", "--limit", "5", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["title"] == "A source-grounded paper"
    assert ("search", {"query": "vision language", "limit": 5}) in _FakeApi.calls


def test_papers_info_prints_structured_metadata(monkeypatch):
    _install_fake(monkeypatch)

    result = CliRunner().invoke(
        mega.app,
        ["papers", "info", "2607.05394", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "2607.05394"
    assert ("info", {"id": "2607.05394"}) in _FakeApi.calls


def test_papers_read_prints_source_grounded_markdown(monkeypatch):
    _install_fake(monkeypatch)

    result = CliRunner().invoke(
        mega.app,
        ["papers", "read", "2607.05394", "--format", "agent"],
    )

    assert result.exit_code == 0, result.output
    assert result.output == "# Test paper\n\n`arXiv:2607.05394`\n"


def test_papers_read_supports_json_output(monkeypatch):
    _install_fake(monkeypatch)

    result = CliRunner().invoke(
        mega.app,
        ["papers", "read", "2607.05394", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "arxiv_id": "2607.05394",
        "markdown": "# Test paper\n\n`arXiv:2607.05394`\n",
    }

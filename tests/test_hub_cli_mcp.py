from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from megatensors._hub.cli import mcp
from megatensors.hub import McpMarketplaceInfo


def _listing(*, available: bool = True) -> McpMarketplaceInfo:
    return McpMarketplaceInfo(
        repo_id="mega/xpuoj",
        title="XPUOJ",
        summary="Contest problems and official judge results.",
        description="README-backed XPUOJ companion.",
        category="developer-tools",
        tags=("xpuoj", "judge"),
        price_per_call=1,
        status="published",
        url="https://mega.tensorplay.cn/mcps/mega/xpuoj",
        endpoint="https://mega.tensorplay.cn/mcp",
        owner_handle="mega",
        owner_display_name="MEGA",
        runtime_kind="official_worker",
        runtime_sdk="cloudflare-worker",
        runtime_stage="RUNNING",
        runtime_available=available,
        calls=3,
        consumers=2,
        earned=3,
        created_at="2026-07-28T00:00:00Z",
        updated_at="2026-07-28T00:00:00Z",
        published_at="2026-07-28T00:00:00Z",
        editable=False,
    )


def test_mcp_search_outputs_marketplace_rows(monkeypatch):
    monkeypatch.setattr(
        "megatensors.hub.MegaHubClient.list_mcp_marketplace",
        lambda self, **kwargs: [_listing()],
    )

    result = CliRunner().invoke(
        mcp.mcp_cli, ["search", "judge", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {
            "repo_id": "mega/xpuoj",
            "title": "XPUOJ",
            "summary": "Contest problems and official judge results.",
            "category": "developer-tools",
            "price_per_call": 1,
            "runtime": "official_worker",
            "available": True,
        }
    ]


def test_mcp_info_outputs_single_listing(monkeypatch):
    monkeypatch.setattr(
        "megatensors.hub.MegaHubClient.get_mcp_marketplace",
        lambda self, repo_id: _listing(),
    )

    result = CliRunner().invoke(
        mcp.mcp_cli, ["info", "mega/xpuoj", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repo_id"] == "mega/xpuoj"
    assert payload["endpoint"] == "https://mega.tensorplay.cn/mcp"
    assert payload["runtime"] == "official_worker"


def test_mcp_install_is_atomic_and_never_executes_publisher_code(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "megatensors.hub.MegaHubClient.get_mcp_marketplace",
        lambda self, repo_id: _listing(),
    )

    def fake_snapshot(self, repo_id, *, local_dir, **kwargs):
        assert kwargs["exclude"] == mcp.MCP_COMPANION_EXCLUDES
        target = Path(local_dir)
        target.mkdir(parents=True)
        (target / "README.md").write_text("# XPUOJ\n", encoding="utf-8")
        (target / "install.sh").write_text(
            "exit 99\n", encoding="utf-8"
        )
        return target

    monkeypatch.setattr(
        "megatensors.hub.MegaHubClient.snapshot_download", fake_snapshot
    )

    result = CliRunner().invoke(
        mcp.mcp_cli,
        ["install", "mega/xpuoj", "--dest", str(tmp_path), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    destination = tmp_path / "mega" / "xpuoj"
    assert (destination / "README.md").read_text(encoding="utf-8") == "# XPUOJ\n"
    assert (destination / "install.sh").is_file()
    manifest = json.loads((destination / mcp.MANAGED_MARKER).read_text())
    assert manifest["repo_id"] == "mega/xpuoj"
    assert json.loads(result.output)["executed"] is False


def test_mcp_install_rejects_path_traversal(tmp_path):
    result = CliRunner().invoke(
        mcp.mcp_cli, ["install", "../xpuoj", "--dest", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert not list(tmp_path.iterdir())

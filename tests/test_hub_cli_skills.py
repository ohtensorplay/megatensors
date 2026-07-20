from click.testing import CliRunner
import pytest

from megatensors._hub.cli import skills
from megatensors._hub.errors import CLIError


def test_skills_preview_is_generated_from_live_mega_tree():
    result = CliRunner().invoke(skills.skills_cli, ["preview"])
    assert result.exit_code == 0, result.output
    assert "name: mega-cli" in result.output
    assert "`mega sandbox create" in result.output
    assert "`mega spaces hot-reload" in result.output
    assert "mega jobs schedule " not in result.output


def test_default_skill_install_is_managed_and_atomic(tmp_path):
    result = CliRunner().invoke(skills.skills_cli, ["add", "--dest", str(tmp_path), "--format", "json"])
    assert result.exit_code == 0, result.output
    installed = tmp_path / "mega-cli"
    assert (installed / "SKILL.md").is_file()
    assert (installed / skills.MANAGED_MARKER).is_file()


@pytest.mark.parametrize("source", ["../secrets", "/absolute/skill", "skills/../secret"])
def test_marketplace_sources_cannot_escape_the_bucket_prefix(source):
    with pytest.raises(CLIError):
        skills._safe_marketplace_source(source)

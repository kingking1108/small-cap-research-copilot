from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_copilot.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ["ingest", "ask", "report", "eval"]:
        assert command in result.stdout


def test_ingest_fails_gracefully_with_no_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WATCHLIST_DIR", str(tmp_path))

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "No PDFs found" in result.stdout

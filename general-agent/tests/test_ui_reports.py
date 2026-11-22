"""The interface offers full-report generation and download."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import workspace  # noqa: E402
from general_agent.report import paths as report_paths  # noqa: E402
from qualitative_helpers import evaluated_run  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _render(**state):
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["flow"] = "advanced"
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    return app


def test_results_page_generates_and_downloads_a_full_report(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(report_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")
    study = evaluated_run(tmp_path, classes=3, n=12)
    run_id = study["run_dir"].name

    app = _render(step=7, run_id=run_id)
    assert not app.exception
    assert not app.error
    labels = [button.label for button in app.button]
    assert any("Generate full report" in label for label in labels)
    assert any("not been run" in info.value for info in app.info)

    next(button for button in app.button
         if "Generate full report" in button.label).click()
    app.run()
    assert not app.error
    downloads = [item.label for item in app.get("download_button")]
    assert any("Download Full Report (PDF)" in label for label in downloads)
    assert any("Full report (Markdown)" in label for label in downloads)


def test_examples_page_offers_case_study_and_comparison_pdfs(tmp_path, monkeypatch):
    comparison = report_paths.PROJECT_ROOT / "comparison" / "report.md"
    if not comparison.exists():
        pytest.skip("the comparison report is not present in this checkout")
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")

    app = _render(page="examples")
    assert not app.exception
    assert not app.error
    labels = [button.label for button in app.button]
    assert any("full report (PDF)" in label for label in labels)
    assert any("comparison report (PDF)" in label for label in labels)

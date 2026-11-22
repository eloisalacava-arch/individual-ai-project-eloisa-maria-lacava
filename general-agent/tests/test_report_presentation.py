"""Report presentation: executive summary, readable labels, portable paths, dedup."""

from __future__ import annotations

import re

from general_agent.evaluate.compare import run_comparison
from general_agent.qualitative.run import report as qualitative_report
from general_agent.report import compose
from qualitative_helpers import evaluated_run, run_qualitative


def _study_with_qualitative(tmp_path, run_id="ui_study"):
    study = evaluated_run(tmp_path, classes=3, n=15, run_id=run_id)
    sidecar, _settings = run_qualitative(
        study, tmp_path, sample={"n": 4, "control_n": 2},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 15})
    assert qualitative_report(study["config_path"], sidecar) == 0
    return study


def test_study_report_has_an_executive_summary(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "#### Executive summary" in markdown
    summary = markdown.split("#### Executive summary", 1)[1].split("## 1.", 1)[0]
    assert "accuracy" in summary.lower()
    assert "disagreements" in summary
    assert "hypotheses" in summary.lower() or "not been run" in markdown


def test_study_report_uses_portable_commands(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "/Users/" not in markdown
    assert f"--run-id {study['run_dir'].name}" in markdown
    assert "uv run python scripts/05_evaluate.py --config" in markdown


def test_illustrative_examples_are_not_duplicated(tmp_path):
    study = _study_with_qualitative(tmp_path)
    markdown, _ = compose.compose_study(study["config_path"])
    section = markdown.split("### Illustrative examples", 1)[1].split("\n## ", 1)[0]
    ids = re.findall(r"- \*\*`([^`]+)`\*\*", section)
    assert ids, "expected at least one illustrative example"
    assert len(ids) == len(set(ids)), f"duplicate examples: {ids}"


def test_study_report_labels_replace_codes(tmp_path):
    study = _study_with_qualitative(tmp_path)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "Mixed or conflicting signals" in markdown
    assert "mixed_or_conflicting_signals" not in markdown


def test_comparison_report_has_summary_and_qualitative_section(tmp_path):
    first = _study_with_qualitative(tmp_path / "a", run_id="study_a")
    second = evaluated_run(tmp_path / "b", classes=3, n=15, run_id="study_b")
    out = tmp_path / "comparison"
    assert run_comparison([first["config_path"], second["config_path"]],
                          output_dir=out) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Executive summary" in report
    assert "## 5. Qualitative findings" in report
    assert "Mixed or conflicting signals" in report
    assert "mixed_or_conflicting_signals" not in report
    assert "/Users/" not in report
    assert first["run_dir"].name in report and second["run_dir"].name in report


def test_comparison_report_without_qualitative_states_it(tmp_path):
    first = evaluated_run(tmp_path / "a", classes=3, n=12, run_id="study_a")
    second = evaluated_run(tmp_path / "b", classes=2, n=12, run_id="study_b")
    out = tmp_path / "comparison"
    assert run_comparison([first["config_path"], second["config_path"]],
                          output_dir=out) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "No study has a qualitative analysis yet" in report
    assert "/Users/" not in report

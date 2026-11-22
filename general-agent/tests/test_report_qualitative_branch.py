"""The qualitative branch of the study report: present vs absent, never invented."""

from __future__ import annotations

from general_agent.qualitative.run import report as qualitative_report
from general_agent.report import compose
from qualitative_helpers import evaluated_run, run_qualitative


def _study_with_qualitative(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15)
    sidecar, settings = run_qualitative(
        study, tmp_path, sample={"n": 4, "control_n": 2},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 15})
    assert qualitative_report(study["config_path"], sidecar) == 0
    return study


def test_qualitative_present_includes_hypotheses_and_control(tmp_path):
    study = _study_with_qualitative(tmp_path)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "AI-generated hypotheses" in markdown
    assert "Contrast with the control sample" in markdown
    assert "Illustrative examples" in markdown
    assert "Qualitative analysis has not been run" not in markdown
    # Coverage and at least one reason, shown with its readable label.
    assert "Selected" in markdown and "Explained" in markdown
    assert "Mixed or conflicting signals" in markdown
    assert "mixed_or_conflicting_signals" not in markdown
    # Selection coverage against the population, and the exploratory disclaimer.
    assert "Selection coverage:" in markdown
    assert "No multiplicity adjustment is applied to these reason contrasts" in markdown


def test_qualitative_absent_has_no_explanations(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "Qualitative analysis has not been run" in markdown
    for token in ("mixed_or_conflicting_signals", "figurative_or_sarcastic_language",
                  "Contrast with the control sample", "Illustrative examples"):
        assert token not in markdown

"""The narrative is traceable, disclaimed, and embeds the statistical report."""

from __future__ import annotations

import pandas as pd

from general_agent.qualitative.run import report
from general_agent.qualitative.settings import output_dir
from qualitative_helpers import evaluated_run, run_qualitative


def test_narrative_is_written_with_disclaimers_and_statistics(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar, settings = run_qualitative(
        study, tmp_path, sample={"n": 6, "control_n": 3},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 20})
    assert report(study["config_path"], sidecar) == 0

    narrative = (output_dir(settings) / "narrative_report.md").read_text(encoding="utf-8")
    assert "AI-generated hypotheses" in narrative
    assert "not confirmed causes" in narrative
    # The statistical half is embedded from stage 07.
    assert "Study report" in narrative
    assert "Confusion matrix" in narrative or "accuracy" in narrative.lower()
    # The control sample is discussed and contrasted.
    assert "Control" in narrative or "control" in narrative
    assert "Contrast with the control sample" in narrative


def test_every_pattern_reason_appears_in_the_narrative(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar, settings = run_qualitative(
        study, tmp_path, sample={"n": 6, "control_n": 3},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 20})
    assert report(study["config_path"], sidecar) == 0
    out_dir = output_dir(settings)
    narrative = (out_dir / "narrative_report.md").read_text(encoding="utf-8")
    from general_agent.qualitative.schema import reason_label
    patterns = pd.read_csv(out_dir / "tables" / "q02_patterns.csv")
    for reason in patterns["reason"].unique():
        assert reason_label(reason) in narrative
        assert reason not in narrative  # codes are replaced by readable labels

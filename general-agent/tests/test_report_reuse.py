"""Reuse: report numbers equal the tables and the stage 07 report; no recalculation."""

from __future__ import annotations

import pandas as pd

from general_agent.config import load_config
from general_agent.evaluate.compare import collect_study, fmt
from general_agent.report import compose
from qualitative_helpers import evaluated_run


def test_collect_study_reads_the_same_numbers_as_the_csv(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15)
    cfg = load_config(study["config_path"])
    row, classes, severity, references, pairs, text = collect_study(cfg)
    t02 = pd.read_csv(cfg["_meta"]["output_dir"] + "/results/t02_accuracy.csv").iloc[0]
    t03 = pd.read_csv(cfg["_meta"]["output_dir"] + "/results/t03_kappa.csv")
    assert row["accuracy"] == float(t02["accuracy"])
    assert row["n"] == int(t02["n"])
    unweighted = t03[t03["metric"].str.contains("unweighted")].iloc[0]
    assert row["kappa_unweighted"] == float(unweighted["value"])
    assert list(classes["class"]) == [entry["name"] for entry in cfg["labels"]["classes"]]


def test_study_report_matches_the_single_study_comparison_report(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15)
    study_md, _ = compose.compose_study(study["config_path"])
    comparison = (study["run_dir"] / "report" / "report.md").read_text(encoding="utf-8")
    accuracy = fmt(pd.read_csv(
        study["run_dir"] / "results" / "t02_accuracy.csv").iloc[0]["accuracy"])
    assert accuracy in study_md
    assert accuracy in comparison
    assert "Study report" in comparison  # the preserved stage 07 heading


def test_narrative_blocks_are_reused_by_both_reports(tmp_path):
    from general_agent.qualitative import narrative
    from general_agent.qualitative.run import report as qualitative_report
    from qualitative_helpers import run_qualitative

    study = evaluated_run(tmp_path, classes=3, n=15)
    sidecar, settings = run_qualitative(
        study, tmp_path, sample={"n": 4, "control_n": 2},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 15})
    assert qualitative_report(study["config_path"], sidecar) == 0

    out_dir = study["run_dir"] / "qualitative"
    blocks = narrative.qualitative_blocks(out_dir, settings)
    assert set(blocks) == {"recurring", "examples", "limitations", "conclusions"}
    study_md, _ = compose.compose_study(study["config_path"])
    # Recurring hypotheses, examples, and qualitative limitations come from the
    # shared builder unchanged. Conclusions are study-specific in the composer.
    assert blocks["recurring"][0] in study_md
    assert blocks["examples"][0] in study_md
    assert blocks["limitations"][0] in study_md

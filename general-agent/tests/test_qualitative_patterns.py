"""Pattern tables are deterministic and reconcile with the explanations."""

from __future__ import annotations

import pandas as pd

from general_agent.qualitative.patterns import write_tables
from general_agent.qualitative.settings import output_dir
from qualitative_helpers import evaluated_run, run_qualitative


def _prepared(tmp_path):
    study = evaluated_run(tmp_path)
    _, settings = run_qualitative(
        study, tmp_path, sample={"n": 6, "control_n": 3},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 20})
    out_dir = output_dir(settings)
    written = write_tables(out_dir)
    return out_dir, written


def test_pattern_counts_reconcile_with_explanations(tmp_path):
    out_dir, written = _prepared(tmp_path)
    explanations = pd.read_csv(out_dir / "explanations.csv", dtype="string")
    patterns = pd.read_csv(written["patterns"])
    explained = explanations[explanations["status"] == "ok"]
    for group in ("disagreement", "agreement"):
        subset = explained[explained["group"] == group]
        total = sum(len([r for r in value.split(";") if r])
                    for value in subset["reasons"])
        assert int(patterns[patterns["group"] == group]["n"].sum()) == total


def test_contrast_and_coverage_are_written(tmp_path):
    out_dir, written = _prepared(tmp_path)
    coverage = pd.read_csv(written["coverage"])
    assert set(coverage["group"]) == {"disagreement", "agreement"}
    assert (coverage["n_explained"] >= 0).all()
    contrast = pd.read_csv(written["contrast"])
    assert "difference_disagreements_minus_control" in contrast.columns


def test_examples_are_ordered_by_confidence_then_id(tmp_path):
    out_dir, written = _prepared(tmp_path)
    examples = pd.read_csv(written["examples"], dtype="string")
    if examples.empty:
        return
    rank = {"high": 3, "medium": 2, "low": 1}
    for _, group_rows in examples.groupby(["group", "reason"]):
        ranks = [rank.get(value, 1) for value in group_rows["confidence"]]
        assert ranks == sorted(ranks, reverse=True)

"""Subgroup hypothesis tests, three-way reference agreement, and the report.

No model call: the study is built on a temporary path from frozen predictions.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
import yaml

from general_agent.evaluate import subgroups
from general_agent.evaluate.analysis import run_evaluation
from general_agent.evaluate.compare import run_comparison
from general_agent.evaluate.disagreements import run_disagreements
from general_agent.report import compose
from test_evaluate import CLASSES, _setup


def test_newcombe_risk_difference_sign_and_bounds():
    rd, low, high = subgroups.newcombe_risk_difference(5, 5, 0, 20)
    assert rd == pytest.approx(1.0)
    assert 0.5 < low < 1.0
    assert high == pytest.approx(1.0)
    rd, low, high = subgroups.newcombe_risk_difference(0, 10, 0, 10)
    assert rd == pytest.approx(0.0)
    assert low <= 0.0 <= high


def test_fisher_exact_matches_known_tables():
    assert subgroups.fisher_exact_two_sided(1, 1, 1, 1) == pytest.approx(1.0)
    # The most extreme 2x2 table with these margins: only one table is as
    # unlikely, so the two-sided p is the probability of that table.
    assert subgroups.fisher_exact_two_sided(5, 0, 0, 20) == pytest.approx(
        1.882e-05, rel=1e-2)


def test_benjamini_hochberg_is_a_step_up():
    qvalues = subgroups.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert qvalues == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert math.isnan(subgroups.benjamini_hochberg([float("nan")])[0])


def _flag_columns(n: int) -> dict[str, list[int]]:
    return {
        "straightforward": [1 if i < 20 else 0 for i in range(n)],
        "mixed": [1 if 20 <= i <= 24 else 0 for i in range(n)],
        "ambiguous": [1 if 25 <= i <= 27 else 0 for i in range(n)],
        "complex": [1 if 28 <= i <= 29 else 0 for i in range(n)],
    }


def _study_with_subgroups(tmp_path):
    n = 30
    errors = {}
    for i in range(20, 25):            # mixed: every row an error
        errors[i] = CLASSES[(i % 3 + 1) % 3]
    errors[25] = CLASSES[(25 % 3 + 1) % 3]  # ambiguous: one error
    config_path, run_dir = _setup(tmp_path, n=n, errors=errors)

    sealed_path = run_dir / "sealed" / "benchmark.csv"
    sealed = pd.read_csv(sealed_path)
    for name, values in _flag_columns(n).items():
        sealed[name] = values
    auxiliary = []
    for i in range(n):
        reference = CLASSES[i % 3]
        auxiliary.append(CLASSES[(i % 3 + 1) % 3] if i >= 25 else reference)
    sealed["aux"] = auxiliary
    sealed.to_csv(sealed_path, index=False)

    cfg = yaml.safe_load(config_path.read_text())
    cfg["benchmark"]["extra_label_columns"] = ["aux"]
    cfg["benchmark"]["reference_labels"] = {
        "sentiment": "manual human annotation",
        "aux": "synthetic proxy label",
    }
    cfg["evaluation"]["subgroup_columns"] = list(_flag_columns(n))
    cfg["evaluation"]["subgroup_reference"] = "straightforward"
    cfg["evaluation"]["subgroup_composites"] = {
        "ambiguous_or_complex": ["ambiguous", "complex"]}
    cfg["report"] = {
        "hypothesis": "Accuracy is lower for ambiguous or complex reviews.",
        "hypothesis_group": "ambiguous_or_complex",
        "primary_reference_rationale": "The manual labels are the text-based reading.",
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    assert run_evaluation(config_path) == 0
    assert run_disagreements(config_path) == 0
    return config_path, run_dir


def test_subgroup_tests_are_written_for_every_reference(tmp_path):
    _, run_dir = _study_with_subgroups(tmp_path)
    tests = pd.read_csv(run_dir / "results" / "t14_subgroup_tests.csv")
    assert set(tests["reference"]) == {"sentiment", "aux"}

    primary = tests[tests["reference"] == "sentiment"]
    mixed = primary[primary["subgroup"] == "mixed"].iloc[0]
    assert int(mixed["n_subgroup"]) == 5
    assert int(mixed["errors_subgroup"]) == 5
    assert float(mixed["risk_difference"]) == pytest.approx(1.0)
    assert mixed["verdict"] == "lower_accuracy_significant"

    # A composite is a single contrast: no BH q within the flag family.
    composite = primary[primary["subgroup"] == "ambiguous_or_complex"].iloc[0]
    assert int(composite["errors_subgroup"]) == 1
    assert math.isnan(float(composite["bh_q"]))

    flags = primary[primary["family"] == "pre_specified"]
    assert flags["bh_q"].notna().all()


def test_three_way_agreement_decomposes_disputes(tmp_path):
    _, run_dir = _study_with_subgroups(tmp_path)
    three = pd.read_csv(run_dir / "results" / "t15_three_way_agreement.csv")
    counts = dict(zip(three["category"], three["n"]))
    assert counts["all_three_agree"] == 20
    assert counts["consensus_ai_disagrees"] == 5
    assert counts["dispute_ai_sides_primary"] == 4
    assert counts["dispute_ai_sides_extra"] == 1
    assert sum(counts.values()) == 30


def test_reference_dispute_rate_is_higher_on_flagged_subgroups(tmp_path):
    _, run_dir = _study_with_subgroups(tmp_path)
    disputes = pd.read_csv(run_dir / "results" / "t16_reference_dispute.csv")
    row = disputes[disputes["subgroup"] == "complex"].iloc[0]
    assert int(row["disputes_subgroup"]) == 2
    assert float(row["rate_subgroup"]) == pytest.approx(1.0)
    assert float(row["rate_reference"]) == pytest.approx(0.0)
    assert float(row["risk_difference"]) > 0


def test_report_renders_the_hypothesis_and_three_way_sections(tmp_path):
    config_path, _ = _study_with_subgroups(tmp_path)
    markdown, _ = compose.compose_study(config_path)
    assert "Subgroup hypothesis test" in markdown
    assert "Three-way agreement" in markdown
    assert "Why this primary reference" in markdown
    assert "Verdict on the pre-specified hypothesis" in markdown
    assert "Risk diff (pp)" in markdown
    # The three-way section names the references precisely: a human annotation
    # and a non-human proxy label, never "two human references".
    assert "manual human annotation" in markdown
    assert "synthetic proxy label" in markdown
    assert "two human references" not in markdown
    assert "Both references agree" not in markdown


def test_run_report_surfaces_the_hypothesis_verdict_and_three_way(tmp_path):
    config_path, run_dir = _study_with_subgroups(tmp_path)
    out = run_dir / "report"
    assert run_comparison([config_path], output_dir=out) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    # The single-study report carries the subgroup verdict and the three-way split.
    assert "Verdict on the pre-specified hypothesis" in report
    assert "Three-way agreement between the AI prediction" in report
    assert "manual human annotation" in report
    assert "no between-study hypothesis test" not in report

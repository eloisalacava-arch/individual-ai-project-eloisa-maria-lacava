"""Regression tests for the reporting defects found in the manual acceptance run.

1. Three unordered classes must not be described as binary.
2. No weakest/most-missed category is named when every recall is identical.
3. Missing report metadata must not render as a blank cell.
4. (Study-name persistence is covered in test_acceptance_new_user.py.)
5. Executive summary is a heading, so the PDF gives it spacing.
"""

from __future__ import annotations

import pandas as pd

from general_agent.config import load_config
from general_agent.evaluate.compare import collect_study
from general_agent.report import compose
from qualitative_helpers import evaluated_run


def _classes_frame(recalls: list[float]) -> pd.DataFrame:
    size = len(recalls)
    return pd.DataFrame({
        "class": ["alpha", "beta", "gamma"][:size],
        "n_reference": [5] * size,
        "n_predicted": [5] * size,
        "precision": [1.0] * size,
        "recall": recalls,
        "recall_ci_low": [0.5] * size,
        "recall_ci_high": [1.0] * size,
        "f1": [1.0] * size,
        "f1_ci_low": [1.0] * size,
        "f1_ci_high": [1.0] * size,
    })


def _row() -> dict:
    return {
        "accuracy": 1.0, "accuracy_ci_low": 0.85, "accuracy_ci_high": 1.0,
        "balanced_accuracy": 1.0, "ba_ci_low": 0.9, "ba_ci_high": 1.0,
        "kappa_unweighted": 1.0, "kappa_ci_low": 1.0, "kappa_ci_high": 1.0,
        "kappa_quadratic": float("nan"), "kappa_q_ci_low": float("nan"),
        "kappa_q_ci_high": float("nan"), "n": 22, "n_classes": 3,
        "n_disagreements": 0, "majority_class": "alpha", "majority_share": 0.364,
        "classes": "alpha / beta / gamma", "title": "t", "task": "x",
        "reference": "r", "seed": 1, "ordered": False,
    }


def test_unordered_three_class_report_is_not_called_binary(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15, ordered=False)
    markdown, cfg = compose.compose_study(study["config_path"])
    assert cfg["labels"]["ordered"] is False
    assert cfg["labels"]["classes"][0]["name"]
    assert len(cfg["labels"]["classes"]) == 3
    assert "binary" not in markdown.lower()
    assert "unordered" in markdown.lower()
    assert "Adjacent` is always zero" not in markdown


def test_no_weakest_category_when_recalls_are_identical(tmp_path):
    classes = _classes_frame([1.0, 1.0, 1.0])
    row = _row()
    summary = "\n".join(compose._executive_summary({}, row, classes, False, None))
    assert "Weakest category" not in summary
    assert "same recall" in summary

    distribution = pd.DataFrame(columns=["class", "difference_ai_minus_human"])
    severity = pd.DataFrame(columns=["bucket", "share"])
    interpretation = "\n".join(compose._interpretation(
        row, classes, severity, distribution, False, 3, False, None))
    assert "Recall ranged" not in interpretation
    assert "recalled equally" in interpretation

    conclusions = "\n".join(compose._conclusions(row, classes, False, None))
    assert "most often missed" not in conclusions
    assert "No category was missed" in conclusions


def test_weakest_category_is_named_only_when_recall_differs():
    classes = _classes_frame([0.5, 1.0, 1.0])
    row = _row()
    summary = "\n".join(compose._executive_summary({}, row, classes, False, None))
    assert "Weakest category: `alpha`" in summary
    conclusions = "\n".join(compose._conclusions(row, classes, False, None))
    assert "most often missed was `alpha`" in conclusions


def test_missing_reference_metadata_is_not_blank(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12, reference_description="")
    cfg = load_config(study["config_path"])
    row, *_ = collect_study(cfg)
    # Derived from the benchmark file name, never blank.
    assert row["reference"].strip()
    assert "Human labels from" in row["reference"]
    assert "benchmark.csv" in row["reference"]
    assert row["task"].strip()
    markdown, _ = compose.compose_study(study["config_path"])
    assert "| Reference |  |" not in markdown
    assert "Human labels from `benchmark.csv`" in markdown
    assert "- Reference: ." not in markdown


def test_executive_summary_is_a_heading_for_spacing(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "#### Executive summary" in markdown
    assert "**Executive summary**" not in markdown

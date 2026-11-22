"""Stage 07: cross-dataset tables, figures, report, and genericity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from general_agent.evaluate.analysis import run_evaluation
from general_agent.evaluate.compare import run_comparison
from general_agent.evaluate.disagreements import run_disagreements
from helpers import make_config


def _study(tmp_path: Path, run_id: str, classes: int, error_offset: int):
    names = ["negative", "neutral", "positive"][:classes]
    rows = [[f"r{i}", f"observation {i} with a sentence of moderate length"]
            for i in range(24)]
    bench = [[f"r{i}", names[i % classes]] for i in range(24)]
    (tmp_path).mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / f"{run_id}_dataset.csv"
    benchmark = tmp_path / f"{run_id}_benchmark.csv"
    pd.DataFrame(rows, columns=["review_id", "review_text"]).to_csv(dataset, index=False)
    pd.DataFrame(bench, columns=["review_id", "sentiment"]).to_csv(benchmark, index=False)

    patch = {
        "run": {"run_id": run_id},
        "dataset": {"path": str(dataset)},
        "benchmark": {"path": str(benchmark)},
        "labels": {"ordered": classes == 3, "classes": [
            {"name": name, "definition": name} for name in names]},
        "evaluation": {"kappa": ["unweighted"] + (["quadratic"] if classes == 3 else []),
                       "bootstrap_draws": 100},
        "report": {"title": run_id, "task_description": f"{classes}-class task",
                   "reference_description": "synthetic reference"},
    }
    config_path = make_config(tmp_path, patch=patch)

    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for i in range(24):
        predicted = names[i % classes]
        if i < error_offset:
            predicted = names[(i + 1) % classes]
        predictions.append({"id": f"r{i}", "label": predicted, "status": "ok",
                            "attempts": 1, "model": "test/model"})
    pd.DataFrame(predictions).to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "sealed").mkdir(exist_ok=True)
    pd.DataFrame(bench, columns=["id", "sentiment"]).to_csv(
        run_dir / "sealed" / "benchmark.csv", index=False)
    (run_dir / "blind").mkdir(exist_ok=True)
    pd.DataFrame([[row[0], row[1]] for row in rows], columns=["id", "text"]).to_csv(
        run_dir / "blind" / "blind.csv", index=False)
    return config_path


def test_single_study_report_is_not_labelled_cross_dataset(tmp_path):
    config_path = _study(tmp_path, "study_only", classes=3, error_offset=2)
    assert run_evaluation(config_path) == 0
    assert run_disagreements(config_path) == 0

    out = tmp_path / "report"
    assert run_comparison([config_path], output_dir=out) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Study report:")
    assert "Cross-dataset comparison" not in report


def test_comparison_builds_tables_figures_and_report(tmp_path):
    first = _study(tmp_path / "a", "study_a", classes=3, error_offset=6)
    second = _study(tmp_path / "b", "study_b", classes=2, error_offset=4)
    for config_path in (first, second):
        assert run_evaluation(config_path) == 0
        assert run_disagreements(config_path) == 0

    out = tmp_path / "comparison"
    assert run_comparison([first, second], output_dir=out) == 0

    for name in ("cross_dataset_metrics.csv", "cross_dataset_classes.csv",
                 "cross_dataset_severity.csv", "cross_dataset_references.csv"):
        assert (out / "tables" / name).exists(), name
    for name in ("fig1_accuracy_kappa.svg", "fig2_class_recall.svg",
                 "fig3_severity.svg"):
        assert (out / "figures" / name).exists(), name

    metrics = pd.read_csv(out / "tables" / "cross_dataset_metrics.csv")
    assert set(metrics["dataset"]) == {"study_a", "study_b"}
    assert set(metrics["n_classes"]) == {2, 3}

    report = (out / "report.md").read_text(encoding="utf-8")
    for fragment in ("study_a", "study_b", "Limitations", "Reproducibility"):
        assert fragment in report
    # Numbers in the report must come from the tables.
    accuracy = metrics.loc[metrics["dataset"] == "study_a", "accuracy"].iloc[0]
    assert f"{accuracy:.3f}" in report


def test_comparison_states_interval_overlap_and_benchmark_differences(tmp_path):
    first = _study(tmp_path / "a", "study_a", classes=3, error_offset=6)
    second = _study(tmp_path / "b", "study_b", classes=2, error_offset=4)
    for config_path in (first, second):
        assert run_evaluation(config_path) == 0
        assert run_disagreements(config_path) == 0

    out = tmp_path / "comparison"
    assert run_comparison([first, second], output_dir=out) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "no between-study hypothesis test was performed" in report
    assert "not directly comparable" in report
    assert "The studies are independent samples" in report


def test_qualitative_section_reports_coverage_and_control_counts():
    from general_agent.evaluate import compare

    metrics = pd.DataFrame([{"dataset": "s1", "n": 30, "n_disagreements": 6}])
    summary = {
        "run_id": "s1", "selected": 10, "explained": 10,
        "selected_disagreement": 6, "selected_agreement": 4,
        "contrast": pd.DataFrame(columns=[
            "reason", "share_disagreements", "share_control",
            "difference_disagreements_minus_control"]),
        "top_disagreement": None, "top_control": None,
    }
    text = "\n".join(compare._qualitative_section([summary], metrics))
    assert "6 of 6 disagreements and 4 of 24 agreements were sampled" in text
    assert "10 of 10 were explained" in text
    assert "No multiplicity adjustment is applied" in text

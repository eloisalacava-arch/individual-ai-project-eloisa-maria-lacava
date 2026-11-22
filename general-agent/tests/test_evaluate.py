"""Stage 05: freeze order, join reconciliation, tables, disagreements vs errors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from general_agent.config import load_config
from general_agent.evaluate import analysis
from general_agent.evaluate.analysis import run_evaluation
from helpers import make_config, write_rows

CLASSES = ["negative", "neutral", "positive"]
RUN_DIR = ("runs", "test_run")


def _setup(tmp_path, n=30, errors=None, drop_prediction_id=None,
           duplicate_prediction_id=False, duplicate_benchmark_id=False,
           failed_prediction_index=None, draws=200):
    rows, bench = [], []
    for index in range(n):
        reference = CLASSES[index % 3]
        rows.append([f"r{index}", f"observation {index} with enough words to look like text"])
        bench.append([f"r{index}", reference])
    if duplicate_benchmark_id:
        bench.append(bench[0])

    dataset_path = tmp_path / "dataset.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    write_rows(dataset_path, ["review_id", "review_text"], rows)
    write_rows(benchmark_path, ["review_id", "sentiment"], bench)

    config_path = make_config(tmp_path, patch={
        "dataset": {"path": str(dataset_path)},
        "benchmark": {"path": str(benchmark_path)},
        "labels": {
            "ordered": True,
            "classes": [
                {"name": "negative", "definition": "unfavourable"},
                {"name": "neutral", "definition": "neither"},
                {"name": "positive", "definition": "favourable"},
            ],
        },
        "evaluation": {"kappa": ["unweighted", "quadratic"], "bootstrap_draws": draws},
    })

    predictions = []
    for index in range(n):
        predicted = CLASSES[index % 3]
        if errors and index in errors:
            predicted = errors[index]
        status = "ok"
        if failed_prediction_index == index:
            predicted = ""
            status = "failed_all_attempts"
        predictions.append({
            "id": f"r{index}", "label": predicted, "status": status,
            "attempts": 1, "model": "test/model",
        })
    if drop_prediction_id is not None:
        predictions = [row for row in predictions if row["id"] != drop_prediction_id]
    if duplicate_prediction_id:
        predictions.append(dict(predictions[0]))

    run_dir = tmp_path.joinpath(*RUN_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(run_dir / "predictions.csv", index=False)

    sealed_dir = run_dir / "sealed"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    write_rows(sealed_dir / "benchmark.csv", ["id", "sentiment"], bench)

    blind_dir = run_dir / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    write_rows(blind_dir / "blind.csv", ["id", "text"],
               [[row[0], row[1]] for row in rows])
    return config_path, run_dir


def _read(run_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(run_dir / "results" / name)


def test_clean_evaluation_writes_every_table_and_known_accuracy(tmp_path):
    # Six adjacent-step errors, so quadratic-weighted kappa should exceed the
    # unweighted one. Opposite-step behaviour is covered in test_metrics.py.
    errors = {0: "neutral", 1: "negative", 2: "neutral",
              3: "neutral", 4: "positive", 5: "neutral"}
    config_path, run_dir = _setup(tmp_path, errors=errors)
    assert run_evaluation(config_path) == 0

    results = run_dir / "results"
    expected = {
        "predictions_freeze.json", "frozen_predictions.csv", "evaluation_log.txt",
        "run_metadata.json", "t01_join_and_sample.csv", "t02_accuracy.csv",
        "t03_kappa.csv", "t04_confusion_matrix.csv", "t05_per_class_metrics.csv",
        "t06_class_distribution.csv", "t07_ordinal_errors.csv",
        "t08_disagreements.csv", "t09_category_summary.csv",
        "t10_error_status.csv",
    }
    assert expected == {path.name for path in results.iterdir() if path.is_file()}

    accuracy = _read(run_dir, "t02_accuracy.csv")
    assert accuracy.loc[0, "n"] == 30
    assert accuracy.loc[0, "n_correct"] == 24
    assert accuracy.loc[0, "accuracy"] == pytest.approx(0.8)
    assert accuracy.loc[0, "accuracy_ci_low"] < 0.8 < accuracy.loc[0, "accuracy_ci_high"]

    kappas = _read(run_dir, "t03_kappa.csv")
    assert set(kappas["metric"]) == {
        "cohen_kappa_unweighted", "cohen_kappa_quadratic_weighted"}
    quadratic = kappas.loc[kappas["metric"].str.contains("quadratic"), "value"].iloc[0]
    unweighted = kappas.loc[kappas["metric"].str.contains("unweighted"), "value"].iloc[0]
    assert quadratic > unweighted

    join = _read(run_dir, "t01_join_and_sample.csv")
    analysed = join.loc[join["step"] == "analysed", "n"].iloc[0]
    assert analysed == 30

    disagreements = _read(run_dir, "t08_disagreements.csv")
    assert len(disagreements) == 6
    assert set(disagreements["error_status"]) == {"unconfirmed_disagreement"}

    ordinal = _read(run_dir, "t07_ordinal_errors.csv")
    buckets = dict(zip(ordinal["bucket"], ordinal["n"]))
    assert buckets["agree"] == 24
    assert buckets["adjacent"] == 6
    assert buckets["opposite"] == 0

    status = _read(run_dir, "t10_error_status.csv")
    confirmed = status.loc[status["status"] == "confirmed_model_error", "n"].iloc[0]
    assert confirmed == 0
    meta = json.loads((results / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["checks"]["status"] == "PASS"


def test_predictions_are_frozen_before_the_benchmark_is_read(tmp_path, monkeypatch):
    config_path, run_dir = _setup(tmp_path)
    results = run_dir / "results"
    original = analysis.load_benchmark

    def guarded(out_dir):
        assert (results / "predictions_freeze.json").exists(), "benchmark opened before freeze"
        assert (results / "frozen_predictions.csv").exists()
        return original(out_dir)

    monkeypatch.setattr(analysis, "load_benchmark", guarded)
    assert run_evaluation(config_path) == 0


def test_frozen_copy_matches_the_source_hash(tmp_path):
    config_path, run_dir = _setup(tmp_path)
    assert run_evaluation(config_path) == 0
    freeze = json.loads((run_dir / "results" / "predictions_freeze.json").read_text())
    assert freeze["source_sha256"] == freeze["frozen_sha256"]
    assert freeze["n_rows"] == 30


def test_missing_prediction_row_is_reported_not_ignored(tmp_path):
    config_path, run_dir = _setup(tmp_path, drop_prediction_id="r7")
    code = run_evaluation(config_path)
    assert code == 1
    join = _read(run_dir, "t01_join_and_sample.csv")
    missing = join.loc[join["step"] == "benchmark rows without a prediction", "n"].iloc[0]
    assert missing == 1
    assert join.loc[join["step"] == "analysed", "n"].iloc[0] == 29
    log = (run_dir / "results" / "evaluation_log.txt").read_text()
    assert "FAIL" in log and "every_prediction_has_a_reference" in log


def test_failed_prediction_is_excluded_and_counted(tmp_path):
    config_path, run_dir = _setup(tmp_path, failed_prediction_index=3)
    code = run_evaluation(config_path)
    assert code == 1  # surfaced as a caveat, not silently dropped
    join = _read(run_dir, "t01_join_and_sample.csv")
    assert join.loc[join["step"] == "excluded failed predictions", "n"].iloc[0] == 1
    assert join.loc[join["step"] == "analysed", "n"].iloc[0] == 29


def test_evaluation_log_reports_the_real_failure_count(tmp_path):
    config_path, run_dir = _setup(tmp_path, failed_prediction_index=3)
    assert run_evaluation(config_path) == 1
    log = (run_dir / "results" / "evaluation_log.txt").read_text()
    assert "1 failed prediction(s)" in log
    assert "0 failed predictions" not in log


def test_evaluation_log_is_dataset_neutral(tmp_path):
    config_path, run_dir = _setup(tmp_path)
    assert run_evaluation(config_path) == 0
    log = (run_dir / "results" / "evaluation_log.txt").read_text()
    assert "75%" not in log
    assert "phrasebank" not in log.lower()
    assert "0 failed predictions" in log


def test_duplicate_prediction_ids_abort_before_benchmark_access(tmp_path):
    config_path, run_dir = _setup(tmp_path, duplicate_prediction_id=True)
    assert run_evaluation(config_path) == 1
    log = (run_dir / "results" / "evaluation_log.txt").read_text()
    assert "ABORT" in log
    assert not (run_dir / "results" / "t02_accuracy.csv").exists()


def test_duplicate_benchmark_ids_abort_the_join(tmp_path):
    config_path, run_dir = _setup(tmp_path, duplicate_benchmark_id=True)
    assert run_evaluation(config_path) == 1
    log = (run_dir / "results" / "evaluation_log.txt").read_text()
    assert "every_prediction_has_a_reference" in log or "benchmark_ids_unique" in log


def test_tables_never_claim_a_confirmed_error(tmp_path):
    config_path, run_dir = _setup(tmp_path, errors={0: "positive"})
    assert run_evaluation(config_path) == 0
    status = _read(run_dir, "t10_error_status.csv")
    row = status.loc[status["status"] == "confirmed_model_error"].iloc[0]
    assert int(row["n"]) == 0
    assert row["confirmable_with_this_data"] in (False, "False")


def test_disagreement_evidence_note_is_reference_neutral(tmp_path):
    # The note must describe whatever primary reference the study declares, not
    # a fixed annotation process (the phrasebank wording used to leak here).
    config_path, run_dir = _setup(tmp_path, errors={0: "positive"})
    assert run_evaluation(config_path) == 0
    note = _read(run_dir, "t08_disagreements.csv")["evidence_note"].iloc[0]
    assert "5-8 annotators" not in note
    assert "75%" not in note
    assert "majority" not in note
    assert "sentiment" in note  # the declared primary label column
    status = _read(run_dir, "t10_error_status.csv")
    unconfirmed = status.loc[
        status["status"] == "unconfirmed_disagreement", "definition"].iloc[0]
    assert unconfirmed == "prediction differs from the primary reference label"

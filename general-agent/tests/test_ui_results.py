"""The results view reads the engine tables and never recomputes a metric."""

from __future__ import annotations

import math

import pandas as pd

from app import engine_adapter, result_view
from general_agent.config import load_config
from stubs import StubProvider
from ui_helpers import make_ui_study

from test_ui_engine_adapter import _seal_and_pilot


def _evaluated_study(tmp_path, classes: int = 3):
    study = make_ui_study(tmp_path, classes=classes, n=15)
    stub = _seal_and_pilot(study)
    assert engine_adapter.approve(study["config_path"]).ok
    assert engine_adapter.classify_batch(study["config_path"], None, provider=stub).ok
    assert engine_adapter.evaluate(study["config_path"]).code in (0, 1)
    assert engine_adapter.disagreements(study["config_path"]).code in (0, 1)
    return study


def test_displayed_accuracy_matches_the_csv(tmp_path):
    study = _evaluated_study(tmp_path)
    frames = result_view.load_results(study["run_dir"])
    displayed = result_view.accuracy_row(frames)
    csv_row = pd.read_csv(study["run_dir"] / "results" / "t02_accuracy.csv").iloc[0]
    assert float(displayed["accuracy"]) == float(csv_row["accuracy"])
    assert int(displayed["n"]) == int(csv_row["n"])


def test_kappa_rows_are_read_not_invented(tmp_path):
    study = _evaluated_study(tmp_path)
    frames = result_view.load_results(study["run_dir"])
    rows = result_view.kappa_rows(frames)
    csv_metrics = set(pd.read_csv(
        study["run_dir"] / "results" / "t03_kappa.csv")["metric"])
    assert {row["metric"] for row in rows} == csv_metrics


def test_confusion_matrix_is_oriented_rows_human_columns_ai(tmp_path):
    study = _evaluated_study(tmp_path)
    matrix, _ = result_view.confusion_frames(study["run_dir"])
    classes = load_config(study["config_path"])["labels"]["classes"]
    names = [entry["name"] for entry in classes]
    assert list(matrix.columns) == names
    assert list(matrix.index) == names
    assert int(matrix.values.sum()) == 15


def test_disagreements_are_joined_to_the_blind_text(tmp_path):
    study = _evaluated_study(tmp_path)
    frame = result_view.text_disagreements(study["run_dir"])
    assert frame is not None
    assert {"id", "reference", "predicted", "text"} <= set(frame.columns)


def test_not_estimable_is_labelled_not_zeroed():
    assert result_view.fmt(float("nan")) == "not estimable"
    assert result_view.fmt(None) == "not estimable"
    assert result_view.fmt(0.1234) == "0.123"
    assert result_view.fmt(0) == "0.000"


def test_download_bundles_include_results_and_exclude_the_seal(tmp_path):
    study = _evaluated_study(tmp_path)
    results_files = result_view.result_files(study["run_dir"])
    assert any(path.name == "t02_accuracy.csv" for path in results_files)
    archive = result_view.make_zip(study["run_dir"], results_files)
    assert len(archive) > 0

    provenance = result_view.provenance_files(study["run_dir"])
    assert provenance
    assert all("sealed" not in path.parts for path in provenance)
    assert any(path.name == "study.yaml" for path in provenance)
    assert any(path.name == "attempts.jsonl" for path in provenance)

"""Run workspaces, upload staging, and table inspection."""

from __future__ import annotations

import pandas as pd

from app import workspace


def test_stage_upload_and_inspect_table(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    staged = workspace.stage_upload(
        "session1", "dataset", "reviews.csv",
        b"id,text\n1,hello world\n2,another row\n")
    assert staged["original_name"] == "reviews.csv"
    meta = workspace.inspect_table(staged["path"])
    assert meta["columns"] == ["id", "text"]
    assert meta["n_rows"] == 2
    assert len(meta["preview"]) == 2


def test_make_id_adds_a_column_without_dropping_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    staged = workspace.stage_upload(
        "s", "dataset", "d.csv", b"text\nalpha\nbeta\n")
    benchmark = workspace.stage_upload(
        "s", "benchmark", "b.csv", b"id,label\n1,positive\n2,negative\n")
    meta = workspace.inspect_table(staged["path"])
    run_dir = tmp_path / "runs" / "study"
    materialised = workspace.materialise_inputs(
        run_dir, staged, benchmark, meta, make_id=True)
    assert materialised["dataset_id_column"] == "observation_id"
    derived = pd.read_csv(run_dir / materialised["dataset_rel"])
    assert len(derived) == 2
    assert list(derived.columns)[0] == "observation_id"
    assert "rows: 2 -> 2" in materialised["row_note"]


def test_next_run_id_never_reuses_a_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    (tmp_path / "runs" / "study").mkdir(parents=True)
    (tmp_path / "runs" / "study-2").mkdir(parents=True)
    assert workspace.next_run_id("study") == "study-3"


def test_list_runs_skips_drafts_and_non_studies(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(workspace, "RUNS_ROOT", runs)
    (runs / "_drafts").mkdir(parents=True)
    (runs / "keep").mkdir()
    (runs / "keep" / "study.yaml").write_text("run:\n  run_id: keep\n", encoding="utf-8")
    (runs / "case").mkdir()
    listed = workspace.list_runs()
    assert [run["run_id"] for run in listed] == ["keep"]

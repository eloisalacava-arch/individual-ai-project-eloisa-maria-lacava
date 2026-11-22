"""Stage 02: sampling determinism, blinding, and sealing."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from general_agent.audit import run_audit
from general_agent.blind import BlindError, reconcile_frames, run_blind, select_indices
from general_agent.config import load_config
from general_agent.sealing import sealed_is_locked
from helpers import make_config, prepare_case, write_rows


def _blind_ids(tmp_path, config_path):
    frame = pd.read_csv(tmp_path / "runs" / "test_run" / "blind" / "blind.csv")
    return frame


def test_blind_file_has_only_id_and_text(tmp_path):
    config_path, _, _ = prepare_case(tmp_path, n=12, sample_n=6)
    assert run_blind(config_path) == 0
    frame = _blind_ids(tmp_path, config_path)
    assert list(frame.columns) == ["id", "text"]
    blind_dir = tmp_path / "runs" / "test_run" / "blind"
    assert [p.name for p in blind_dir.iterdir()] == ["blind.csv"]


def test_sample_size_is_honoured_and_deterministic(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_config, _, _ = prepare_case(first_dir, n=20, sample_n=8)
    second_config, _, _ = prepare_case(second_dir, n=20, sample_n=8)
    assert run_blind(first_config) == 0
    assert run_blind(second_config) == 0

    first_ids = list(_blind_ids(first_dir, first_config)["id"])
    second_ids = list(_blind_ids(second_dir, second_config)["id"])
    assert len(first_ids) == 8
    assert first_ids == second_ids


def test_null_sample_size_uses_every_row(tmp_path):
    config_path, _, _ = prepare_case(tmp_path, n=10, sample_n=None)
    assert run_blind(config_path) == 0
    assert len(_blind_ids(tmp_path, config_path)) == 10


def test_unique_text_reduces_the_draw_when_duplicates_exist(tmp_path):
    config_path, _, _ = prepare_case(tmp_path, n=12, sample_n=12, duplicate_pairs=6)
    assert run_blind(config_path) == 0
    frame = _blind_ids(tmp_path, config_path)
    assert len(frame) == 6
    assert frame["text"].nunique() == 6
    manifest = json.loads(
        (tmp_path / "runs" / "test_run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stages"]["02_blind"]["sample"]["requested_n"] == 12
    assert manifest["stages"]["02_blind"]["sample"]["selected_n"] == 6


def test_selection_is_independent_of_the_label_column(tmp_path):
    dataset = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(30)],
        "review_text": [f"observation number {i}" for i in range(30)],
        "label": ["positive"] * 30,
    })
    mapping = {"id_column": "review_id", "text_column": "review_text"}
    config = {"n": 10, "unique_text": True, "method": "random"}
    first = select_indices(dataset, mapping, config, seed=20261116)

    dataset["label"] = list(reversed(dataset["label"]))
    second = select_indices(dataset, mapping, config, seed=20261116)
    assert first == second


def test_unstratified_sample_keeps_both_classes(tmp_path):
    rows = [[f"r{i}", f"observation {i} with enough words to look like text"]
            for i in range(100)]
    labels = ["positive"] * 70 + ["negative"] * 30
    dataset_path = tmp_path / "imbalanced_dataset.csv"
    benchmark_path = tmp_path / "imbalanced_benchmark.csv"
    write_rows(dataset_path, ["review_id", "review_text"], rows)
    write_rows(benchmark_path, ["review_id", "sentiment"],
               [[row[0], label] for row, label in zip(rows, labels)])

    config_path = make_config(tmp_path, patch={
        "dataset": {"path": str(dataset_path)},
        "benchmark": {"path": str(benchmark_path)},
        "sample": {"n": 40},
    })
    from general_agent.audit import run_audit
    assert run_audit(config_path) == 0
    assert run_blind(config_path) == 0

    sealed = pd.read_csv(tmp_path / "runs" / "test_run" / "sealed" / "benchmark.csv")
    counts = sealed["sentiment"].value_counts()
    assert set(counts.index) == {"positive", "negative"}
    positive_share = counts["positive"] / len(sealed)
    assert 0.55 <= positive_share <= 0.85  # natural 70%, no stratification applied


def test_sealed_file_is_separate_and_restricted_to_the_sample(tmp_path):
    config_path, _, _ = prepare_case(tmp_path, n=20, sample_n=7)
    assert run_blind(config_path) == 0
    sealed_path = tmp_path / "runs" / "test_run" / "sealed" / "benchmark.csv"
    blind_path = tmp_path / "runs" / "test_run" / "blind" / "blind.csv"
    assert sealed_path.exists()
    assert sealed_path.parent != blind_path.parent
    sealed = pd.read_csv(sealed_path)
    blind = pd.read_csv(blind_path)
    assert len(sealed) == 7
    assert list(sealed["id"]) == list(blind["id"])
    assert "sentiment" in sealed.columns


def test_blind_is_shuffled_not_file_order(tmp_path):
    config_path, dataset_path, _ = prepare_case(tmp_path, n=20, sample_n=20)
    assert run_blind(config_path) == 0
    source_order = list(pd.read_csv(dataset_path)["review_id"])
    blind_order = list(_blind_ids(tmp_path, config_path)["id"])
    assert sorted(blind_order) == sorted(source_order)
    assert blind_order != source_order


def test_manifest_records_hashes_and_sample(tmp_path):
    config_path, _, _ = prepare_case(tmp_path, n=12, sample_n=5)
    assert run_blind(config_path) == 0
    manifest = json.loads(
        (tmp_path / "runs" / "test_run" / "manifest.json").read_text(encoding="utf-8")
    )
    stage = manifest["stages"]["02_blind"]
    assert stage["rows_selected"] == 5
    assert stage["sample"]["stratified"] is False
    assert len(stage["blind_sha256"]) == 64
    assert len(stage["sealed_sha256"]) == 64
    assert sealed_is_locked(tmp_path / "runs" / "test_run") is False


def test_blind_refuses_without_an_audit(tmp_path):
    config_path = make_config(tmp_path)
    with pytest.raises(BlindError) as excinfo:
        run_blind(config_path)
    assert "audit" in str(excinfo.value).lower()

def test_inner_join_policy_is_materialised_by_blind(tmp_path):
    """Stage 02 must take the ID intersection the audit only checked in memory."""
    config_path = make_config(
        tmp_path,
        benchmark="benchmark_subset.csv",
        patch={"benchmark": {"on_id_mismatch": "inner_join"}},
    )
    assert run_audit(config_path) == 0
    assert run_blind(config_path) == 0

    blind = _blind_ids(tmp_path, config_path)
    sealed = pd.read_csv(tmp_path / "runs" / "test_run" / "sealed" / "benchmark.csv")
    assert sorted(blind["id"]) == ["r1", "r2"]
    assert sorted(sealed["id"]) == ["r1", "r2"]
    assert list(sealed["id"]) == list(blind["id"])

    manifest = json.loads(
        (tmp_path / "runs" / "test_run" / "manifest.json").read_text(encoding="utf-8")
    )
    stage = manifest["stages"]["02_blind"]
    steps = [entry["step"] for entry in stage["reconciliation"]]
    assert "dataset restricted to matched IDs" in steps
    assert "benchmark restricted to matched IDs" in steps
    assert stage["rows_dataset_after_policies"] == 2
    assert stage["rows_benchmark_after_policies"] == 2
    assert stage["rows_in"] == 4  # the raw read is still reported


def test_unknown_labels_are_excluded_by_blind_with_counts(tmp_path):
    """A label excluded by the audit must also leave the blind input."""
    rows = [[f"r{i}", f"observation {i} with enough words to look like real text"]
            for i in range(4)]
    bench = [["r0", "positive"], ["r1", "negative"],
             ["r2", "positive"], ["r3", "mystery"]]
    dataset_path = tmp_path / "dataset_unknown.csv"
    benchmark_path = tmp_path / "benchmark_unknown.csv"
    write_rows(dataset_path, ["review_id", "review_text"], rows)
    write_rows(benchmark_path, ["review_id", "sentiment"], bench)

    config_path = make_config(tmp_path, patch={
        "dataset": {"path": str(dataset_path)},
        "benchmark": {"path": str(benchmark_path), "on_unknown_label": "exclude"},
    })
    assert run_audit(config_path) == 0
    assert run_blind(config_path) == 0

    blind = _blind_ids(tmp_path, config_path)
    sealed = pd.read_csv(tmp_path / "runs" / "test_run" / "sealed" / "benchmark.csv")
    assert sorted(blind["id"]) == ["r0", "r1", "r2"]
    assert sorted(sealed["id"]) == ["r0", "r1", "r2"]
    assert "mystery" not in set(sealed["sentiment"])

    manifest = json.loads(
        (tmp_path / "runs" / "test_run" / "manifest.json").read_text(encoding="utf-8")
    )
    stage = manifest["stages"]["02_blind"]
    steps = [entry["step"] for entry in stage["reconciliation"]]
    assert "benchmark rows without a usable primary label" in steps
    assert "dataset restricted to scorable IDs" in steps
    assert stage["rows_dataset_after_policies"] == 3
    assert stage["rows_benchmark_after_policies"] == 3


def test_reconcile_refuses_a_mismatch_under_the_error_policy(tmp_path):
    cfg = {
        "benchmark": {"on_id_mismatch": "error", "on_unknown_label": "error"},
        "labels": {"classes": [
            {"name": "positive", "definition": "favourable"},
            {"name": "negative", "definition": "unfavourable"},
        ]},
    }
    mapping = {"id_column": "review_id", "id_column_benchmark": "review_id",
               "label_column": "sentiment"}
    dataset = pd.DataFrame({"review_id": ["r1", "r2"], "text": ["a", "b"]})
    benchmark = pd.DataFrame({"review_id": ["r1"], "sentiment": ["positive"]})
    with pytest.raises(BlindError) as excinfo:
        reconcile_frames(cfg, mapping, dataset, benchmark, lambda *args: None)
    assert "on_id_mismatch" in str(excinfo.value)


def test_rerunning_audit_or_configure_preserves_later_stage_records(tmp_path):
    from general_agent.audit import run_audit
    from general_agent.cli import main

    config_path, _, _ = prepare_case(tmp_path, n=8, sample_n=4)
    assert run_blind(config_path) == 0
    assert run_audit(config_path) == 0
    assert main(["configure", "--config", str(config_path)]) == 0

    manifest = json.loads(
        (tmp_path / "runs" / "test_run" / "manifest.json").read_text(encoding="utf-8")
    )
    stages = manifest["stages"]
    assert "02_blind" in stages
    assert "01_audit" in stages
    assert "00_configure" in stages
    # Re-running configure must not drop the audit's top-level totals.
    assert manifest["checks"]["status"] == "PASS"
    assert manifest["row_counts"]["dataset_analysed"] == 8

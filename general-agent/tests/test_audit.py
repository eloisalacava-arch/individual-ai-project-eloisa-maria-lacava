"""Stage 01 audit: status codes, artefacts, row accounting, leakage failure."""

from __future__ import annotations

import json

from general_agent.audit import run_audit
from helpers import make_config


def _manifest(tmp_path, run_id="test_run"):
    path = tmp_path / "runs" / run_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_clean_audit_passes_and_writes_three_artefacts(tmp_path):
    code = run_audit(make_config(tmp_path))
    assert code == 0
    out = tmp_path / "runs" / "test_run"
    for name in ("decisions.json", "audit.txt", "manifest.json"):
        assert (out / name).exists(), name
    manifest = _manifest(tmp_path)
    assert manifest["checks"]["status"] == "PASS"
    assert manifest["checks"]["failed"] == []
    assert manifest["row_counts"]["dataset_analysed"] == 4
    assert manifest["row_counts"]["benchmark_analysed"] == 4
    decisions = json.loads((out / "decisions.json").read_text(encoding="utf-8"))
    assert decisions["status"] == "resolved"
    assert decisions["resolved_mapping"]["label_column"] == "sentiment"
    assert decisions["decisions"]


def test_label_leak_in_dataset_fails_the_audit(tmp_path):
    code = run_audit(make_config(tmp_path, dataset="dataset_leak.csv"))
    assert code == 1
    manifest = _manifest(tmp_path)
    assert "dataset_excludes_reference_label_columns" in manifest["checks"]["failed"]
    assert manifest["checks"]["status"] == "FAIL"


def test_ambiguous_numeric_labels_need_clarification(tmp_path):
    path = make_config(
        tmp_path,
        benchmark="benchmark_numeric.csv",
        patch={"benchmark": {"label_column": None}},
    )
    code = run_audit(path)
    assert code == 2
    manifest = _manifest(tmp_path)
    assert manifest["checks"]["status"] == "needs_clarification"
    decisions = json.loads(
        (tmp_path / "runs" / "test_run" / "decisions.json").read_text(encoding="utf-8")
    )
    assert decisions["questions"]
    assert any(q["field"] == "benchmark.label_map" for q in decisions["questions"])


def test_inner_join_records_dropped_rows_with_reasons(tmp_path):
    path = make_config(
        tmp_path,
        benchmark="benchmark_subset.csv",
        patch={"benchmark": {"on_id_mismatch": "inner_join"}},
    )
    code = run_audit(path)
    assert code == 0
    manifest = _manifest(tmp_path)
    accounting = manifest["row_accounting"]
    assert len(accounting) == 2
    assert accounting[0]["step"] == "dataset restricted to matched IDs"
    assert accounting[0]["rows_before"] == 4
    assert accounting[0]["rows_after"] == 2
    assert accounting[0]["rows_dropped"] == 2
    assert "reason" in accounting[0] and accounting[0]["reason"]
    assert manifest["row_counts"]["dataset_analysed"] == 2


def test_missing_input_fails_without_crashing(tmp_path):
    path = make_config(
        tmp_path, patch={"dataset": {"path": str(tmp_path / "absent.csv")}}
    )
    code = run_audit(path)
    assert code == 1
    assert "ABORT" in (tmp_path / "runs" / "test_run" / "audit.txt").read_text(
        encoding="utf-8"
    )


def test_manifest_never_contains_a_key_and_records_model_identity(tmp_path):
    run_audit(make_config(tmp_path))
    manifest = _manifest(tmp_path)
    text = json.dumps(manifest)
    assert "test/model" in text
    assert manifest["model"]["provider"] == "opencode_cli"
    assert manifest["model"]["temperature_status"] == "not_requested"
    assert "seed" in manifest and manifest["seed"] == 20261116
    assert manifest["inputs"]["dataset"]["sha256"]
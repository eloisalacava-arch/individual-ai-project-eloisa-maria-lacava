"""Guards for the simple pipeline: reliable seal detection, benign re-seal,
concurrency, and seal-before-model-call.

No network or CLI call is made: every provider is a stub.
"""

from __future__ import annotations

import json
import os
import threading

import pandas as pd

from app import engine_adapter, pipeline, workspace
from general_agent.config import load_config
from general_agent.sealing import sealed_is_locked
from stubs import StubProvider
from ui_helpers import make_ui_study


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    from general_agent.report import paths as report_paths
    monkeypatch.setattr(report_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")


def _authorization(cfg_path, *, qualitative: bool = False) -> dict:
    cfg = load_config(cfg_path)
    return {
        "run_id": cfg["run"]["run_id"],
        "config_sha256": cfg["_meta"]["config_sha256"],
        "classification": {"max_calls": cfg["budget"]["max_calls"],
                           "max_usd": 1.0},
        "qualitative": {"enabled": qualitative},
        "batch_size": 25,
        "allow_unclean_pilot": False,
    }


def _seal_artifacts(run_dir, *, marker: bool = True):
    """Write the artefacts a completed seal leaves behind."""
    (run_dir / "blind").mkdir(exist_ok=True)
    (run_dir / "blind" / "blind.csv").write_text(
        "id,text\nr0,hello\n", encoding="utf-8")
    (run_dir / "sealed").mkdir(exist_ok=True)
    (run_dir / "sealed" / "benchmark.csv").write_text(
        "id,label\nr0,positive\n", encoding="utf-8")
    if marker:
        (run_dir / "blind_public_info.json").write_text("{}", encoding="utf-8")


def _resolved_decisions(run_dir):
    (run_dir / "decisions.json").write_text(
        json.dumps({"status": "resolved"}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fix A: sealed detection must not depend on a locked directory
# --------------------------------------------------------------------------- #

def test_stage_flags_see_sealed_when_the_sealed_dir_is_locked(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    run_dir = tmp_path / "runs" / "locked"
    run_dir.mkdir(parents=True)
    _seal_artifacts(run_dir)
    os.chmod(run_dir / "sealed", 0o000)
    try:
        assert not (run_dir / "sealed" / "benchmark.csv").exists()
        flags = workspace.stage_flags("locked")
        assert flags["sealed"] is True
    finally:
        os.chmod(run_dir / "sealed", 0o755)


def test_missing_marker_means_not_sealed(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    run_dir = tmp_path / "runs" / "no_marker"
    run_dir.mkdir(parents=True)
    _seal_artifacts(run_dir, marker=False)
    assert workspace.stage_flags("no_marker")["sealed"] is False


# --------------------------------------------------------------------------- #
# Fix B: an already-sealed run is benign, not a failure
# --------------------------------------------------------------------------- #

def test_pipeline_skips_seal_when_the_marker_exists(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="already", classes=2, n=8)
    run_dir = study["run_dir"]
    _resolved_decisions(run_dir)
    _seal_artifacts(run_dir)

    calls = {"seal": 0}

    def fake_seal(*args, **kwargs):
        calls["seal"] += 1
        raise AssertionError("seal must not be called when sealed")

    monkeypatch.setattr(engine_adapter, "seal", fake_seal)
    monkeypatch.setattr(
        engine_adapter, "pilot",
        lambda *a, **k: engine_adapter.StageResult(
            stage="pilot", code=0, log="", error=None, kind="ok"))

    advance = pipeline.advance("already", study["config_path"],
                               _authorization(study["config_path"]))
    assert advance.status == "working"
    assert advance.stage == "pilot"
    assert calls["seal"] == 0


def test_pipeline_recovers_when_seal_reports_already_sealed(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="benign", classes=2, n=8)
    run_dir = study["run_dir"]
    _resolved_decisions(run_dir)
    # The seal artefacts exist but the marker is missing, so the pipeline tries
    # to seal. The engine reports "already sealed" and writes the marker.
    _seal_artifacts(run_dir, marker=False)

    def already_sealed(*args, **kwargs):
        (run_dir / "blind_public_info.json").write_text("{}", encoding="utf-8")
        return engine_adapter.StageResult(
            stage="seal", code=1, log="already sealed", error="already sealed",
            kind="failed")

    monkeypatch.setattr(engine_adapter, "seal", already_sealed)
    advance = pipeline.advance("benign", study["config_path"],
                               _authorization(study["config_path"]))
    assert advance.status == "working"
    assert advance.stage == "seal"
    assert "already sealed" in advance.action.lower()


# --------------------------------------------------------------------------- #
# Resuming a classified run must not re-run classification
# --------------------------------------------------------------------------- #

def test_resume_classified_run_goes_to_evaluation(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="resume", classes=2, n=8)
    run_dir = study["run_dir"]
    _resolved_decisions(run_dir)
    _seal_artifacts(run_dir)
    (run_dir / "pilot").mkdir(exist_ok=True)
    pd.DataFrame([{"id": "r0", "label": "positive", "status": "ok"}]).to_csv(
        run_dir / "pilot" / "pilot_predictions.csv", index=False)
    (run_dir / "pilot" / "approval.json").write_text("{}", encoding="utf-8")
    pd.DataFrame([{"id": "r0", "label": "positive", "status": "ok"}]).to_csv(
        run_dir / "predictions.csv", index=False)

    calls = {"pilot": 0, "classify": 0, "evaluate": 0}

    def fake_pilot(*args, **kwargs):
        calls["pilot"] += 1
        return engine_adapter.StageResult(stage="pilot", code=0, log="",
                                          error=None, kind="ok")

    def fake_classify(*args, **kwargs):
        calls["classify"] += 1
        return engine_adapter.StageResult(stage="classify", code=0, log="",
                                          error=None, kind="ok")

    monkeypatch.setattr(engine_adapter, "classify_batch", fake_classify)

    def fake_evaluate(*args, **kwargs):
        calls["evaluate"] += 1
        (run_dir / "results").mkdir(exist_ok=True)
        (run_dir / "results" / "t02_accuracy.csv").write_text(
            "accuracy\n1.0\n", encoding="utf-8")
        return engine_adapter.StageResult(stage="evaluate", code=0, log="",
                                          error=None, kind="ok")

    monkeypatch.setattr(engine_adapter, "pilot", fake_pilot)
    monkeypatch.setattr(engine_adapter, "evaluate", fake_evaluate)

    advance = pipeline.advance("resume", study["config_path"],
                               _authorization(study["config_path"]))
    assert advance.status == "working"
    assert advance.stage == "evaluate"
    assert calls == {"pilot": 0, "classify": 0, "evaluate": 1}


# --------------------------------------------------------------------------- #
# Fix D: the per-run guard prevents overlap and releases on error
# --------------------------------------------------------------------------- #

def test_concurrent_advance_returns_busy(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="busy", classes=2, n=8)
    _resolved_decisions(study["run_dir"])
    _seal_artifacts(study["run_dir"])
    auth = _authorization(study["config_path"])
    seen = {}

    def worker():
        seen["advance"] = pipeline.advance("busy", study["config_path"], auth)

    with pipeline._run_lock("busy") as held:
        assert held
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
    assert seen["advance"].status == "busy"


def test_lock_releases_after_a_failed_stage(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="release", classes=2, n=8)
    auth = _authorization(study["config_path"])

    monkeypatch.setattr(
        engine_adapter, "audit",
        lambda *a, **k: engine_adapter.StageResult(
            stage="audit", code=1, log="boom", error="boom", kind="failed"))

    first = pipeline.advance("release", study["config_path"], auth)
    assert first.status == "error"
    second = pipeline.advance("release", study["config_path"], auth)
    assert second.status == "error"  # not "busy": the lock was released


# --------------------------------------------------------------------------- #
# Seal-before-model-call invariant
# --------------------------------------------------------------------------- #

def test_seal_precedes_every_model_call(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="order", classes=2, n=10)
    run_dir = study["run_dir"]
    auth = _authorization(study["config_path"])
    observed = []

    def on_call(provider, prompt):
        observed.append({
            "marker": (run_dir / "blind_public_info.json").exists(),
            "locked": sealed_is_locked(run_dir),
        })

    provider = StubProvider(load_config(study["config_path"]), label="positive",
                            on_call=on_call)
    for _ in range(30):
        advance = pipeline.advance("order", study["config_path"], auth,
                                   classification_provider=provider)
        if advance.status in {"done", "error", "blocked", "stopped"}:
            break
    assert advance.status == "done", advance.detail
    assert observed, "the provider was never called"
    assert all(item["marker"] for item in observed)
    assert all(item["locked"] for item in observed)

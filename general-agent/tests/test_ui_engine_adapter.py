"""The UI adapter drives the existing stages and maps their exit codes."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app import engine_adapter, workspace
from general_agent.config import load_config
from stubs import StubProvider
from ui_helpers import make_ui_study


def _seal_and_pilot(study, stub=None):
    config_path = study["config_path"]
    assert engine_adapter.configure(config_path).ok
    assert engine_adapter.audit(config_path).ok
    assert engine_adapter.seal(config_path).ok
    stub = stub or StubProvider(load_config(config_path), label=study["classes"][0])
    assert engine_adapter.pilot(config_path, provider=stub).ok
    return stub


def test_full_ui_flow_reaches_a_generated_report(tmp_path):
    study = make_ui_study(tmp_path, classes=3)
    config_path = study["config_path"]
    stub = _seal_and_pilot(study)

    assert engine_adapter.approve(config_path).ok
    classified = engine_adapter.classify_batch(config_path, None, provider=stub)
    assert classified.ok, classified.log

    evaluated = engine_adapter.evaluate(config_path)
    assert evaluated.code in (0, 1), evaluated.log
    assert engine_adapter.disagreements(config_path).code in (0, 1)
    report = engine_adapter.report(config_path, study["run_dir"] / "report")
    assert report.ok, report.log

    report_text = (study["run_dir"] / "report" / "report.md").read_text(encoding="utf-8")
    assert report_text.startswith("# Study report:")
    assert (study["run_dir"] / "results" / "t02_accuracy.csv").exists()


def test_full_run_requires_an_approved_pilot(tmp_path):
    study = make_ui_study(tmp_path)
    config_path = study["config_path"]
    assert engine_adapter.configure(config_path).ok
    assert engine_adapter.audit(config_path).ok
    assert engine_adapter.seal(config_path).ok

    result = engine_adapter.classify_batch(config_path, None,
                                           provider=StubProvider(
                                               load_config(config_path)))
    assert result.code == 2
    assert result.kind == "precondition"


def test_budget_cap_leaves_rows_pending_and_is_resumable(tmp_path):
    study = make_ui_study(tmp_path, n=12, max_calls=8)
    config_path = study["config_path"]
    stub = _seal_and_pilot(study, StubProvider(load_config(config_path),
                                               label=study["classes"][0]))
    assert engine_adapter.approve(config_path).ok

    stopped = engine_adapter.classify_batch(config_path, None, provider=stub)
    assert stopped.code == 3
    frame = pd.read_csv(study["run_dir"] / "predictions.csv", dtype="string")
    assert (frame["status"] == "pending").sum() > 0
    assert workspace.calls_used(study["run_dir"].name) <= 8


def test_a_fatal_provider_error_stops_without_imputing(tmp_path):
    study = make_ui_study(tmp_path, n=6, max_calls=40)
    config_path = study["config_path"]
    assert engine_adapter.configure(config_path).ok
    assert engine_adapter.audit(config_path).ok
    assert engine_adapter.seal(config_path).ok

    stub = StubProvider(load_config(config_path), status="blocked_auth")
    result = engine_adapter.pilot(config_path, provider=stub)
    assert result.code == 3
    frame = pd.read_csv(study["run_dir"] / "pilot" / "pilot_predictions.csv",
                        dtype="string")
    assert (frame["status"] == "blocked_auth").any()
    assert (frame["label"] == "").all()  # never imputed


def test_classifier_never_opens_the_benchmark(tmp_path, monkeypatch):
    import general_agent.classify as classify_module

    opened: list[str] = []
    original = classify_module.read_table

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(classify_module, "read_table", spy)

    study = make_ui_study(tmp_path, n=6)
    stub = _seal_and_pilot(study, StubProvider(load_config(study["config_path"]),
                                               label=study["classes"][0]))
    assert engine_adapter.approve(study["config_path"]).ok
    assert engine_adapter.classify_batch(study["config_path"], None,
                                         provider=stub).ok
    assert all("benchmark" not in path and "sealed" not in path for path in opened), opened


def test_approval_binds_the_model_and_prompt(tmp_path):
    study = make_ui_study(tmp_path, n=6)
    config_path = study["config_path"]
    _seal_and_pilot(study)
    assert engine_adapter.approve(config_path).ok
    approved, detail = engine_adapter.approval_status(config_path)
    assert approved, detail
    assert "approved at" in detail


def test_configure_audit_seal_artifacts_exist(tmp_path):
    study = make_ui_study(tmp_path, n=6)
    _seal_and_pilot(study)
    run_dir = study["run_dir"]
    for relative in ("manifest.json", "decisions.json", "audit.txt",
                     "blind/blind.csv", "sealed/benchmark.csv",
                     "pilot/pilot_predictions.csv"):
        assert (run_dir / relative).exists(), relative
    blind = pd.read_csv(run_dir / "blind" / "blind.csv")
    assert list(blind.columns) == ["id", "text"]

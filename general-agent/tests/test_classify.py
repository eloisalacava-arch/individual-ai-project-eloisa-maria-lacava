"""Stages 03-04: approval gate, resume, budget, fatal stop, atomic saves."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from general_agent.blind import run_blind
from general_agent.classify import ClassificationError, approve_pilot, run
from general_agent.config import load_config
from general_agent.sealing import sealed_is_locked
from helpers import prepare_case
from stubs import StubProvider

RUN_DIR = ("runs", "test_run")


def _setup(tmp_path, **kwargs):
    config_path, _, _ = prepare_case(tmp_path, **kwargs)
    assert run_blind(config_path) == 0
    return config_path


def _stub(config_path, **kwargs):
    return StubProvider(load_config(config_path), **kwargs)


def _run_dir(tmp_path):
    return tmp_path.joinpath(*RUN_DIR)


def test_full_run_requires_an_approved_pilot(tmp_path):
    config_path = _setup(tmp_path, n=8, pilot_n=3)
    code = run(config_path, mode="full", provider=_stub(config_path))
    assert code == 2
    assert not (_run_dir(tmp_path) / "predictions.csv").exists()


def test_pilot_then_approval_then_full(tmp_path):
    config_path = _setup(tmp_path, n=10, pilot_n=3)

    pilot_stub = _stub(config_path)
    assert run(config_path, mode="pilot", provider=pilot_stub) == 0
    pilot = pd.read_csv(_run_dir(tmp_path) / "pilot" / "pilot_predictions.csv")
    assert len(pilot) == 3
    assert set(pilot["status"]) == {"ok"}
    assert len(pilot_stub.calls) == 3

    report = json.loads(
        (_run_dir(tmp_path) / "pilot" / "pilot_report.json").read_text(encoding="utf-8")
    )
    assert report["n_ok"] == 3
    assert report["prompt_template_sha"]

    assert approve_pilot(config_path, note="pilot looks fine") == 0
    approval = json.loads(
        (_run_dir(tmp_path) / "pilot" / "approval.json").read_text(encoding="utf-8")
    )
    assert approval["n_pilot"] == 3
    assert approval["note"] == "pilot looks fine"

    full_stub = _stub(config_path)
    assert run(config_path, mode="full", provider=full_stub) == 0
    predictions = pd.read_csv(_run_dir(tmp_path) / "predictions.csv")
    assert len(predictions) == 10
    assert set(predictions["status"]) == {"ok"}
    assert len(full_stub.calls) == 10


def test_resume_skips_rows_already_ok(tmp_path):
    config_path = _setup(tmp_path, n=8, pilot_n=2)
    run(config_path, mode="pilot", provider=_stub(config_path))
    approve_pilot(config_path)

    first = _stub(config_path)
    assert run(config_path, mode="full", limit=4, provider=first) == 1
    assert len(first.calls) == 4
    partial = pd.read_csv(_run_dir(tmp_path) / "predictions.csv")
    assert len(partial) == 8  # the file always holds every row
    assert (partial["status"] == "ok").sum() == 4
    assert (partial["status"] == "pending").sum() == 4

    second = _stub(config_path)
    assert run(config_path, mode="full", resume=True, provider=second) == 0
    assert len(second.calls) == 4
    final = pd.read_csv(_run_dir(tmp_path) / "predictions.csv")
    assert set(final["status"]) == {"ok"}
    assert final["attempts"].astype(int).sum() == 8


def test_budget_cap_stops_and_is_resumable(tmp_path):
    config_path = _setup(tmp_path, n=8, pilot_n=5, max_calls=2)
    stub = _stub(config_path)
    assert run(config_path, mode="pilot", provider=stub) == 3
    assert len(stub.calls) == 2
    pilot = pd.read_csv(_run_dir(tmp_path) / "pilot" / "pilot_predictions.csv")
    assert (pilot["status"] == "ok").sum() == 2
    assert (pilot["status"] == "pending").sum() == 3

    # A second invocation must not exceed the same cap.
    second = _stub(config_path)
    assert run(config_path, mode="pilot", resume=True, provider=second) == 3
    assert len(second.calls) == 0


def test_fatal_provider_error_stops_the_run(tmp_path):
    config_path = _setup(tmp_path, n=8, pilot_n=4)
    stub = _stub(config_path, status="blocked_billing")
    assert run(config_path, mode="pilot", provider=stub) == 3
    assert len(stub.calls) == 1
    pilot = pd.read_csv(_run_dir(tmp_path) / "pilot" / "pilot_predictions.csv")
    assert pilot.loc[0, "status"] == "blocked_billing"
    assert (pilot["status"] == "pending").sum() == 3


def test_sealed_directory_is_locked_during_calls_and_unlocked_after(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    out_dir = _run_dir(tmp_path)
    observed = []

    def probe(provider, prompt):
        observed.append(sealed_is_locked(out_dir))

    stub = _stub(config_path, on_call=probe)
    assert run(config_path, mode="pilot", provider=stub) == 0
    assert observed and all(observed)
    assert sealed_is_locked(out_dir) is False


def test_manifest_records_provider_and_budget_without_temperature_claims(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    run(config_path, mode="pilot", provider=_stub(config_path))
    manifest = json.loads((_run_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    stage = manifest["stages"]["03_pilot"]
    assert stage["provider"]["temperature_status"] == "not_requested"
    assert stage["provider"]["model"] == "test/model"
    assert stage["calls_used_total"] == 2
    assert stage["stopped_reason"] is None
    assert "not billing data" in stage["estimated_usd_note"]


def test_full_run_refuses_after_the_prompt_changed(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    run(config_path, mode="pilot", provider=_stub(config_path))
    approve_pilot(config_path)

    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("favourable overall stance", "a changed definition"), encoding="utf-8"
    )
    assert run(config_path, mode="full", provider=_stub(config_path)) == 2


def test_full_run_refuses_after_the_model_changed(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    run(config_path, mode="pilot", provider=_stub(config_path))
    approve_pilot(config_path)

    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace("test/model", "other/model"), encoding="utf-8")
    assert run(config_path, mode="full", provider=_stub(config_path)) == 2


def test_attempt_log_contains_prompt_output_and_model(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    stub = _stub(config_path)
    run(config_path, mode="pilot", provider=stub)

    log = _run_dir(tmp_path) / "pilot" / "pilot_attempts.jsonl"
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    first = records[0]
    for key in ("prompt", "raw_stdout", "model", "attempt", "ts_start", "status",
                "prompt_sha256", "prompt_template_sha256", "temperature_status"):
        assert key in first
    assert first["status"] == "ok"
    assert first["model"] == "test/model"


def test_repeated_full_run_without_resume_is_refused(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    run(config_path, mode="pilot", provider=_stub(config_path))
    approve_pilot(config_path)
    assert run(config_path, mode="full", provider=_stub(config_path)) == 0
    with pytest.raises(ClassificationError) as excinfo:
        run(config_path, mode="full", provider=_stub(config_path))
    assert "--resume" in str(excinfo.value)

def test_full_run_refuses_when_the_pilot_file_changed_after_approval(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    run(config_path, mode="pilot", provider=_stub(config_path))
    approve_pilot(config_path)
    # Re-running the pilot with a different outcome replaces the approved file.
    run(config_path, mode="pilot", provider=_stub(config_path, label="negative"))
    assert run(config_path, mode="full", provider=_stub(config_path)) == 2


def test_pilot_calls_count_toward_the_total_budget(tmp_path):
    config_path = _setup(tmp_path, n=8, pilot_n=2, max_calls=5)
    pilot_stub = _stub(config_path)
    assert run(config_path, mode="pilot", provider=pilot_stub) == 0
    assert len(pilot_stub.calls) == 2
    approve_pilot(config_path)

    full_stub = _stub(config_path)
    # 2 pilot calls + only 3 remaining, while 8 rows are pending.
    assert run(config_path, mode="full", provider=full_stub) == 3
    assert len(full_stub.calls) == 3
    predictions = pd.read_csv(_run_dir(tmp_path) / "predictions.csv")
    assert (predictions["status"] == "ok").sum() == 3


def test_manifest_records_provider_reported_cost_and_tokens(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    stub = _stub(config_path, cost_usd=0.001,
                 usage={"input": 100, "output": 4, "reasoning": 0,
                        "cache_read": 0, "cache_write": 0})
    assert run(config_path, mode="pilot", provider=stub) == 0
    manifest = json.loads((_run_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    stage = manifest["stages"]["03_pilot"]
    assert stage["actual_usd"] == pytest.approx(0.002)
    assert stage["token_totals"]["input"] == 200
    assert stage["token_totals"]["output"] == 8


def test_noop_resume_does_not_reset_cost_totals(tmp_path):
    config_path = _setup(tmp_path, n=6, pilot_n=2)
    stub = _stub(config_path, cost_usd=0.001,
                 usage={"input": 100, "output": 4, "reasoning": 0,
                        "cache_read": 0, "cache_write": 0})
    assert run(config_path, mode="pilot", provider=stub) == 0
    approve_pilot(config_path)
    assert run(config_path, mode="full", provider=stub) == 0

    manifest = json.loads((_run_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    stage = manifest["stages"]["04_classify"]
    assert stage["actual_usd"] == pytest.approx(0.008)
    assert stage["token_totals"]["input"] == 800

    # A no-op resume makes no calls and must not overwrite the totals with zero.
    assert run(config_path, mode="full", resume=True, provider=_stub(config_path)) == 0
    refreshed = json.loads((_run_dir(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert refreshed["stages"]["04_classify"]["actual_usd"] == pytest.approx(0.008)
    assert refreshed["stages"]["04_classify"]["token_totals"]["input"] == 800
    assert refreshed["stages"]["04_classify"]["calls_full"] == 6


def test_totals_are_derived_from_raw_output_when_cost_was_not_logged(tmp_path):
    from general_agent.classify import _totals_from_logs
    from general_agent.config import load_config

    config_path, _, _ = prepare_case(tmp_path, n=4, pilot_n=2)
    cfg = load_config(config_path)
    cfg["budget"]["price_per_million"] = {"input": 0.036, "output": 0.5}
    run_dir = _run_dir(tmp_path)
    log = run_dir / "pilot" / "pilot_attempts.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        "prompt": "x" * 400,
        "raw_stdout": '{"type":"step_finish","part":{"type":"step-finish",'
                      '"cost":0.0001,"tokens":{"input":10,"output":2,'
                      '"reasoning":0,"cache":{"read":5,"write":0}}}}\n',
        "status": "ok",
    }
    log.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    actual, estimated, tokens = _totals_from_logs(run_dir, cfg)
    assert actual == pytest.approx(0.0001)
    assert tokens["input"] == 10
    assert tokens["cache_read"] == 5
    assert estimated > 0

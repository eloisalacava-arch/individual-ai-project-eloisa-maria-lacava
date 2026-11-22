"""The real three-step UI acceptance run (live model calls) completed correctly.

Unlike ``tests/test_ui_simple.py`` (which injects a stub provider), this module
inspects the committed evidence of the live run ``runs/simple_interface_test``:
22/22 parseable predictions, calls and spend inside the authorized caps, and a
completed evaluation and study report. It skips when the run is not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs" / "simple_interface_test"
REPORT = PROJECT / "reports" / "simple_interface_test"


def _require() -> None:
    if not (RUN / "manifest.json").exists() or not (RUN / "ui_authorization.json").exists():
        pytest.skip("simple_interface_test acceptance run is not present in this checkout")


def test_run_has_exactly_22_valid_predictions():
    _require()
    predictions = pd.read_csv(RUN / "predictions.csv", dtype="string")
    assert len(predictions) == 22
    assert set(predictions["status"]) == {"ok"}
    # One attempt-log line per classified row in the full run.
    assert (RUN / "logs" / "attempts.jsonl").read_text(
        encoding="utf-8").count("\n") == 22


def test_calls_and_spend_stay_within_the_authorized_caps():
    _require()
    authorization = json.loads((RUN / "ui_authorization.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    pilot = manifest["stages"]["03_pilot"]
    classify = manifest["stages"]["04_classify"]

    # Authorized ceilings, as recorded at authorization time.
    assert authorization["classification"]["max_calls"] == classify["max_calls"] == 32
    assert authorization["total_max_calls"] == 62

    assert classify["calls_used_total"] == 27
    assert classify["calls_used_total"] <= authorization["classification"]["max_calls"]
    assert classify["calls_used_total"] <= authorization["total_max_calls"]
    assert classify["actual_usd"] <= authorization["classification"]["max_usd"]
    # Pilot (5) plus the full run (22) are the 27 calls counted against the cap.
    assert pilot["calls_used_total"] + 22 == classify["calls_used_total"] == 27


def test_evaluation_and_study_report_completed():
    _require()
    assert (RUN / "results" / "t02_accuracy.csv").exists()
    assert (REPORT / "study_report.md").exists()
    assert (REPORT / "study_report.pdf").exists()


def test_qualitative_was_skipped_because_there_were_no_disagreements():
    # The authorization enabled qualitative explanations, but every prediction
    # matched the reference, so the pipeline correctly ran no qualitative calls.
    _require()
    accuracy = pd.read_csv(RUN / "results" / "t02_accuracy.csv").iloc[0]
    assert int(accuracy["n_correct"]) == int(accuracy["n"]) == 22
    assert not (RUN / "qualitative" / "explanations.csv").exists()
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    assert "08_qualitative" not in (manifest.get("stages") or {})

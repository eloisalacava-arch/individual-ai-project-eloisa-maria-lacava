"""Manifest call accounting: caps respected, legacy inconsistencies documented.

Two questions are separated deliberately:

* **Budget violations** — a stage whose recorded ``calls_used_total`` exceeds its
  recorded ``max_calls``, or whose ``actual_usd`` exceeds the run's ``max_usd``.
  There are none.
* **Schema inconsistencies** — the ``03_pilot`` records for the two original case
  studies were written by an older code path that left ``calls_pilot`` and
  ``calls_full`` at 0 while ``calls_used_total`` held the true count (5). Those
  historical counts are preserved, not rewritten, and are listed below. Any new
  mismatch fails the test.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"

# Written by an older code path; documented in VERIFICATION.md.
LEGACY_SPLIT_MISMATCH = {
    ("imdb_case_study", "03_pilot"),
    ("phrasebank_75agree", "03_pilot"),
}


def _manifests():
    for path in sorted(RUNS.glob("*/manifest.json")):
        try:
            yield path.parent.name, json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue


def _stages(manifest: dict):
    for stage, record in (manifest.get("stages") or {}).items():
        if isinstance(record, dict):
            yield stage, record


def test_no_stage_exceeds_its_call_cap():
    violations = []
    for run_id, manifest in _manifests():
        for stage, record in _stages(manifest):
            used, cap = record.get("calls_used_total"), record.get("max_calls")
            if used is not None and cap is not None and used > cap:
                violations.append((run_id, stage, used, cap))
    assert not violations, f"call cap exceeded: {violations}"


def test_no_stage_exceeds_the_run_spend_cap():
    violations = []
    for run_id, manifest in _manifests():
        cap = (manifest.get("budget") or {}).get("max_usd")
        if cap is None:
            continue
        for stage, record in _stages(manifest):
            actual = record.get("actual_usd")
            if actual is not None and actual > cap:
                violations.append((run_id, stage, actual, cap))
    assert not violations, f"spend cap exceeded: {violations}"


def test_split_call_fields_match_total_except_documented_legacy_records():
    undocumented = []
    for run_id, manifest in _manifests():
        for stage, record in _stages(manifest):
            if {"calls_pilot", "calls_full", "calls_used_total"} <= set(record):
                total = record["calls_pilot"] + record["calls_full"]
                if total != record["calls_used_total"] \
                        and (run_id, stage) not in LEGACY_SPLIT_MISMATCH:
                    undocumented.append(
                        (run_id, stage, total, record["calls_used_total"]))
    assert not undocumented, f"undocumented split mismatch: {undocumented}"


def test_legacy_pilot_records_are_unchanged():
    """The historical counts are preserved, not rewritten to hide the split gap."""
    for run_id, stage in sorted(LEGACY_SPLIT_MISMATCH):
        path = RUNS / run_id / "manifest.json"
        if not path.exists():
            continue
        record = (json.loads(path.read_text(encoding="utf-8")).get("stages")
                  or {}).get(stage)
        if not isinstance(record, dict):
            continue
        assert record.get("calls_used_total") == 5, (run_id, stage)
        assert record.get("calls_pilot") == 0, (run_id, stage)
        assert record.get("calls_full") == 0, (run_id, stage)

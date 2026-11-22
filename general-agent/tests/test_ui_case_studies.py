"""The interface reads the shipped case studies and never writes to them."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app import result_view, workspace

CASES = ["imdb_case_study", "phrasebank_75agree"]


def _hash_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_case_study_results_are_read_only(tmp_path):
    comparison = workspace.PROJECT_ROOT / "comparison"
    before = {run_id: _hash_tree(workspace.run_dir(run_id)) for run_id in CASES}
    before["comparison"] = _hash_tree(comparison)
    if not all(before[run_id] for run_id in CASES):
        pytest.skip("case-study runs are not present in this checkout")

    # The examples page only reads through these functions.
    workspace.list_runs()
    for run_id in CASES:
        frames = result_view.load_results(workspace.run_dir(run_id))
        assert "t02_accuracy.csv" in frames
        result_view.confusion_frames(workspace.run_dir(run_id))

    after = {run_id: _hash_tree(workspace.run_dir(run_id)) for run_id in CASES}
    after["comparison"] = _hash_tree(comparison)
    assert before == after


def test_case_study_accuracy_is_present():
    for run_id in CASES:
        results = workspace.run_dir(run_id) / "results" / "t02_accuracy.csv"
        if not results.exists():
            pytest.skip(f"{run_id} has no evaluated results")
        frame = result_view.load_results(workspace.run_dir(run_id))
        row = result_view.accuracy_row(frame)
        assert row["n"] > 0

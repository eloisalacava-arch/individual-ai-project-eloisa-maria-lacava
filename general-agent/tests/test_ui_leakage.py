"""Benchmark values must never leave the sealed directory in a UI-driven run."""

from __future__ import annotations

import pandas as pd

from app import engine_adapter
from general_agent.config import load_config
from general_agent.prompts import build_prompt
from stubs import StubProvider
from ui_helpers import make_ui_study

from test_ui_engine_adapter import _seal_and_pilot

MARKER = "UI-LEAK-MARKER-4b71e2"


def test_poison_marker_stays_inside_the_seal(tmp_path):
    study = make_ui_study(tmp_path, n=8, marker=MARKER,
                          subgroup_columns=["human_notes"])
    stub = _seal_and_pilot(study)
    assert engine_adapter.approve(study["config_path"]).ok
    assert engine_adapter.classify_batch(study["config_path"], None, provider=stub).ok
    assert engine_adapter.evaluate(study["config_path"]).code in (0, 1)

    run_dir = study["run_dir"]
    sealed = pd.read_csv(run_dir / "sealed" / "benchmark.csv", dtype="string")
    assert MARKER in set(sealed["human_notes"])  # the marker is real

    # ``inputs/`` holds the user's own uploaded benchmark copy and ``sealed/``
    # holds the sealed benchmark; both are the source of truth, not a leak. The
    # classifier only ever reads blind/blind.csv (asserted separately), so the
    # marker must not appear in any prompt, log, prediction, result, or manifest.
    allowed = {"sealed", "inputs"}
    leaked = []
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.parts[len(run_dir.parts)] in allowed:
            continue
        if path.suffix in {".csv", ".json", ".jsonl", ".txt"}:
            if MARKER in path.read_text(encoding="utf-8", errors="replace"):
                leaked.append(str(path.relative_to(run_dir)))
    assert not leaked, f"marker leaked into: {leaked}"


def test_prompt_preview_contains_only_the_blind_text(tmp_path):
    study = make_ui_study(tmp_path, n=6, marker=MARKER,
                          subgroup_columns=["human_notes"])
    _seal_and_pilot(study)
    run_dir = study["run_dir"]
    blind = pd.read_csv(run_dir / "blind" / "blind.csv", dtype="string")
    assert list(blind.columns) == ["id", "text"]
    cfg = load_config(study["config_path"])
    prompt = build_prompt(cfg, blind["text"].iloc[0])
    assert MARKER not in prompt
    assert blind["text"].iloc[0] in prompt

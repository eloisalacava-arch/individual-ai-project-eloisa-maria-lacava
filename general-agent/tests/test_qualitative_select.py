"""Stage 08 selection: seeded, deterministic, and gated on the freeze."""

from __future__ import annotations

import json

import pytest

from general_agent.config import load_config
from general_agent.qualitative.select import (
    build_selection, require_frozen, write_selection,
)
from general_agent.qualitative.settings import QualitativeError, output_dir
from qualitative_helpers import evaluated_run, qualitative_settings, write_sidecar


def test_selection_is_deterministic_and_seeded(tmp_path):
    study = evaluated_run(tmp_path)
    settings = qualitative_settings(study)
    cfg = load_config(study["config_path"])
    first, accounting = build_selection(study["run_dir"], cfg, settings)
    second, _ = build_selection(study["run_dir"], cfg, settings)
    assert list(first["id"]) == list(second["id"])
    assert set(first["group"]) <= {"disagreement", "agreement"}
    assert (first["text"].str.len() > 0).all()
    assert accounting


def test_selection_respects_the_configured_sizes(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = write_sidecar(study, tmp_path,
                            sample={"n": 3, "control_n": 2},
                            budget={"max_calls": 20})
    settings = qualitative_settings(study, sidecar)
    cfg = load_config(study["config_path"])
    selected, _ = build_selection(study["run_dir"], cfg, settings)
    assert int((selected["group"] == "disagreement").sum()) <= 3
    assert int((selected["group"] == "agreement").sum()) <= 2


def test_control_can_be_disabled(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = write_sidecar(study, tmp_path, sample={"n": 3, "control_n": 0},
                            budget={"max_calls": 20})
    settings = qualitative_settings(study, sidecar)
    cfg = load_config(study["config_path"])
    selected, _ = build_selection(study["run_dir"], cfg, settings)
    assert (selected["group"] == "agreement").sum() == 0


def test_require_frozen_refuses_before_stage_05(tmp_path):
    with pytest.raises(QualitativeError):
        require_frozen(tmp_path)


def test_selection_meta_and_row_accounting_are_written(tmp_path):
    study = evaluated_run(tmp_path)
    settings = qualitative_settings(study)
    cfg = load_config(study["config_path"])
    selected, accounting = build_selection(study["run_dir"], cfg, settings)
    out_dir = output_dir(settings)
    write_selection(out_dir, selected, accounting, settings)
    assert (out_dir / "selected.csv").exists()
    meta = json.loads((out_dir / "selection_meta.json").read_text(encoding="utf-8"))
    assert meta["n_selected"] == len(selected)
    assert meta["signature"]
    assert (out_dir / "tables" / "q00_selection.csv").exists()

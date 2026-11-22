"""Stage 08 pilot gate, budget cap, fatal errors, and resume."""

from __future__ import annotations

import pandas as pd
import pytest

from general_agent.qualitative.run import approve, check_approval, report, run
from general_agent.qualitative.select import require_frozen
from general_agent.qualitative.settings import QualitativeError, output_dir
from qualitative_helpers import evaluated_run, qualitative_settings, write_sidecar
from stubs import JsonStubProvider


def _sidecar(study, tmp_path, **overrides):
    return write_sidecar(study, tmp_path, **overrides)


def test_pilot_then_approve_then_full_run(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = _sidecar(study, tmp_path, sample={"n": 4, "control_n": 2},
                       pilot={"disagreements": 2, "agreements": 1},
                       budget={"max_calls": 12})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings)

    assert run(study["config_path"], sidecar, mode="pilot", provider=provider) == 0
    assert (output_dir(settings) / "pilot" / "pilot_explanations.csv").exists()
    assert approve(study["config_path"], sidecar) == 0
    approved, detail = check_approval(output_dir(settings), settings)
    assert approved, detail
    assert run(study["config_path"], sidecar, mode="full", provider=provider) == 0

    explanations = pd.read_csv(output_dir(settings) / "explanations.csv",
                               dtype="string")
    assert (explanations["status"] == "ok").all()
    assert report(study["config_path"], sidecar) == 0
    assert (output_dir(settings) / "narrative_report.md").exists()


def test_full_run_seeds_ok_pilot_rows_and_skips_them(tmp_path):
    import json

    study = evaluated_run(tmp_path)
    sidecar = _sidecar(study, tmp_path, sample={"n": 4, "control_n": 2},
                       pilot={"disagreements": 2, "agreements": 1},
                       budget={"max_calls": 15})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings)
    assert run(study["config_path"], sidecar, mode="pilot", provider=provider) == 0
    assert approve(study["config_path"], sidecar) == 0

    provider.calls.clear()
    assert run(study["config_path"], sidecar, mode="full", provider=provider) == 0

    out_dir = output_dir(settings)
    full = pd.read_csv(out_dir / "explanations.csv", dtype="string")
    pilot = pd.read_csv(out_dir / "pilot" / "pilot_explanations.csv", dtype="string")
    assert len(provider.calls) == 3          # 3 pilot rows reused, 3 new calls
    assert len(full) == 6
    assert (full["status"] == "ok").all()
    seeded = full[full["id"].isin(pilot["id"])].set_index("id")
    expected = pilot.set_index("id")
    for identifier in seeded.index:
        assert seeded.loc[identifier, "summary"] == expected.loc[identifier, "summary"]
    manifest = json.loads((study["run_dir"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["08_qualitative"]["seeded_from_pilot"] == 3


def test_full_run_requires_an_approved_pilot(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = _sidecar(study, tmp_path, sample={"n": 3, "control_n": 1})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings)
    assert run(study["config_path"], sidecar, mode="full", provider=provider) == 2


def test_approval_is_invalidated_by_a_model_change(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = _sidecar(study, tmp_path, sample={"n": 3, "control_n": 1})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings)
    assert run(study["config_path"], sidecar, mode="pilot", provider=provider) == 0
    assert approve(study["config_path"], sidecar) == 0

    changed = write_sidecar(
        study, tmp_path,
        sample={"n": 3, "control_n": 1},
        model={"provider": "opencode_cli", "name": "other/model",
               "max_attempts": 2})
    changed_settings = qualitative_settings(study, changed)
    approved, detail = check_approval(output_dir(changed_settings), changed_settings)
    assert not approved
    assert "model" in detail


def test_a_budget_cap_stops_and_leaves_rows_pending(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = _sidecar(
        study, tmp_path, sample={"n": 2, "control_n": 0},
        pilot={"disagreements": 1, "agreements": 0},
        budget={"max_calls": 4},
        model={"provider": "opencode_cli", "name": "test/model", "max_attempts": 2})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings, status="api_error")

    assert run(study["config_path"], sidecar, mode="pilot", provider=provider) == 1
    assert approve(study["config_path"], sidecar) == 0
    code = run(study["config_path"], sidecar, mode="full", provider=provider)
    assert code == 3
    frame = pd.read_csv(output_dir(settings) / "explanations.csv", dtype="string")
    assert (frame["status"] == "pending").sum() > 0


def test_a_fatal_provider_error_stops_without_inventing_text(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = _sidecar(study, tmp_path, sample={"n": 3, "control_n": 0},
                       pilot={"disagreements": 1, "agreements": 0},
                       budget={"max_calls": 6})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings, status="blocked_auth")
    assert run(study["config_path"], sidecar, mode="pilot", provider=provider) == 3
    frame = pd.read_csv(output_dir(settings) / "pilot" / "pilot_explanations.csv",
                        dtype="string")
    assert (frame["status"] == "blocked_auth").any()
    assert (frame["summary"] == "").all()


def test_predictions_and_results_are_unchanged(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = _sidecar(study, tmp_path, sample={"n": 3, "control_n": 0},
                       pilot={"disagreements": 1, "agreements": 0},
                       budget={"max_calls": 6})
    settings = qualitative_settings(study, sidecar)
    predictions = study["run_dir"] / "predictions.csv"
    before = predictions.read_bytes()
    provider = JsonStubProvider(settings)
    assert run(study["config_path"], sidecar, mode="pilot", provider=provider) == 0
    assert predictions.read_bytes() == before
    assert not (study["run_dir"] / "results" / "q01_explanations.csv").exists()


def test_run_refuses_before_the_freeze(tmp_path):
    from ui_helpers import make_ui_study
    study = make_ui_study(tmp_path, n=6)
    sidecar = write_sidecar(study, tmp_path, sample={"n": 2, "control_n": 0},
                            pilot={"disagreements": 1, "agreements": 0},
                            budget={"max_calls": 3})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings)
    with pytest.raises(QualitativeError):
        run(study["config_path"], sidecar, mode="pilot", provider=provider)

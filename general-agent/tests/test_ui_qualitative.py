"""The qualitative section: sidecar building, adapter wrappers, headless render."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import engine_adapter, qualitative_support, registry, workspace  # noqa: E402
from general_agent.qualitative.settings import load_settings  # noqa: E402
from qualitative_helpers import (  # noqa: E402
    evaluated_run, qualitative_settings, run_qualitative, write_sidecar,
)
from stubs import JsonStubProvider  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _render(**state):
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["flow"] = "advanced"
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    return app


def test_sidecar_builds_valid_settings(tmp_path):
    study = evaluated_run(tmp_path)
    provider = registry.enabled_providers()[0]
    model = provider["models"][0]
    sidecar = qualitative_support.build_sidecar(provider, model, 5, 3,
                                                20261116, 20, 0.5)
    path = qualitative_support.write_sidecar(study["run_dir"], sidecar)
    settings = load_settings(study["config_path"], path)
    assert settings["sample"]["n"] == 5
    assert settings["sample"]["control_n"] == 3
    assert settings["model"]["max_attempts"] == 2
    assert settings["_meta"]["classes"]
    assert qualitative_support.minimum_calls(5, 3) == 8
    assert qualitative_support.estimate_floor_usd(model, 10) >= 0.0


def test_adapter_runs_the_qualitative_flow(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar = write_sidecar(study, tmp_path, sample={"n": 4, "control_n": 2},
                            pilot={"disagreements": 2, "agreements": 1},
                            budget={"max_calls": 15})
    settings = qualitative_settings(study, sidecar)
    provider = JsonStubProvider(settings)

    assert engine_adapter.qualitative_pilot(
        study["config_path"], sidecar, provider=provider).ok
    assert engine_adapter.qualitative_approve(study["config_path"], sidecar).ok
    approved, detail = engine_adapter.qualitative_approval_status(
        study["config_path"], sidecar)
    assert approved, detail
    assert engine_adapter.qualitative_run(
        study["config_path"], sidecar, provider=provider).ok
    assert engine_adapter.qualitative_report(study["config_path"], sidecar).ok
    assert (study["run_dir"] / "qualitative" / "narrative_report.md").exists()


def test_results_page_shows_the_qualitative_pilot_button(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    study = evaluated_run(tmp_path)
    app = _render(step=7, run_id=study["run_dir"].name)
    assert not app.exception
    labels = [button.label for button in app.button]
    assert any("pilot" in label.lower() for label in labels)
    assert any("Qualitative" in markdown.value for markdown in app.markdown)


def test_results_page_offers_the_narrative_after_a_qualitative_run(tmp_path,
                                                                   monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    study = evaluated_run(tmp_path)
    sidecar, _settings = run_qualitative(
        study, tmp_path, sample={"n": 4, "control_n": 2},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 15})
    assert engine_adapter.qualitative_report(study["config_path"], sidecar).ok

    app = _render(step=7, run_id=study["run_dir"].name)
    assert not app.exception
    downloads = [item.label for item in app.get("download_button")]
    assert any("Narrative" in label for label in downloads)
    assert any("Explanations" in label for label in downloads)

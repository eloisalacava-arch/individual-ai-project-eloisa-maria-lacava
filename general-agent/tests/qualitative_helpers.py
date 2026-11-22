"""Build an evaluated run and its qualitative settings for tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from general_agent.audit import run_audit
from general_agent.blind import run_blind
from general_agent.classify import approve_pilot, run as classify_run
from general_agent.config import load_config
from general_agent.evaluate.analysis import run_evaluation
from general_agent.evaluate.compare import run_comparison
from general_agent.evaluate.disagreements import run_disagreements
from general_agent.qualitative.settings import load_settings
from stubs import StubProvider
from ui_helpers import make_ui_study


def evaluated_run(tmp_path: Path, classes: int = 3, n: int = 15,
                  label: str | None = None, run_id: str = "ui_study",
                  ordered: bool | None = None,
                  reference_description: str = "synthetic reference") -> dict:
    """A study classified, frozen, evaluated, and analysed for disagreements."""
    study = make_ui_study(tmp_path, run_id=run_id, classes=classes, n=n,
                          ordered=ordered,
                          reference_description=reference_description)
    config_path = study["config_path"]
    assert run_audit(config_path) == 0
    assert run_blind(config_path) == 0
    stub = StubProvider(load_config(config_path),
                        label=label or study["classes"][0])
    assert classify_run(config_path, mode="pilot", provider=stub) == 0
    assert approve_pilot(config_path) == 0
    assert classify_run(config_path, mode="full", provider=stub) == 0
    assert run_evaluation(config_path) == 0
    assert run_disagreements(config_path) == 0
    assert run_comparison([config_path],
                          output_dir=study["run_dir"] / "report") == 0
    return study


def write_sidecar(study: dict, tmp_path: Path, **overrides) -> Path:
    """Write a minimal, valid qualitative sidecar next to the study."""
    run_dir = study["run_dir"]
    settings = {
        "study_config": {"path": "../study.yaml"},
        "sample": {"n": 20, "control_n": 5, "method": "random", "seed": 20261116},
        "pilot": {"disagreements": 2, "agreements": 1},
        "budget": {"max_calls": 40, "max_usd": 1.0,
                   "price_per_million": {"input": 0.0, "output": 0.0}},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key] = {**settings[key], **value}
        else:
            settings[key] = value
    directory = run_dir / "qualitative"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return path


def qualitative_settings(study: dict, sidecar: Path | None = None) -> dict:
    return load_settings(study["config_path"], sidecar)


def run_qualitative(study: dict, tmp_path: Path, provider=None, **overrides):
    """Run pilot, approve, and the full explanation run. Returns (sidecar, settings)."""
    from general_agent.qualitative.run import approve, run
    from stubs import JsonStubProvider

    sidecar = write_sidecar(study, tmp_path, **overrides)
    settings = load_settings(study["config_path"], sidecar)
    provider = provider or JsonStubProvider(settings)
    assert run(study["config_path"], sidecar, mode="pilot", provider=provider) == 0
    assert approve(study["config_path"], sidecar) == 0
    assert run(study["config_path"], sidecar, mode="full", provider=provider) == 0
    return sidecar, settings

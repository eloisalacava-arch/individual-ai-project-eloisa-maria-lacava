"""P3: the single-authorized pipeline orchestrator (stub providers only)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from app import pipeline, qualitative_support, registry, workspace
from general_agent.config import load_config
from general_agent.providers.base import Provider, ProviderResult
from general_agent.report import paths as report_paths
from stubs import JsonStubProvider, StubProvider
from ui_helpers import make_ui_study

TERMINAL = {"done", "stopped", "error", "blocked", "needs_pilot_review"}


class FlakyProvider(Provider):
    """Fails to answer for the first N calls, then behaves normally."""

    name = "flaky"
    supports_temperature = False

    def __init__(self, cfg, fail_first: int, label: str = "positive"):
        super().__init__(cfg)
        self.fail_first = fail_first
        self.label = label
        self.calls = 0

    def call(self, prompt, timeout_s):
        self.calls += 1
        if self.calls <= self.fail_first:
            return ProviderResult(status="ok", text="no answer line here",
                                  raw_stdout="flaky",
                                  extra={"parse_method": "flaky"})
        return ProviderResult(status="ok", text=f"ANSWER: {self.label}",
                              raw_stdout="flaky",
                              extra={"parse_method": "flaky"})


class EchoProvider(Provider):
    """Returns the class encoded in the observation, so labels can match exactly."""

    name = "echo"
    supports_temperature = False

    def call(self, prompt, timeout_s):
        match = re.search(r"\[(negative|neutral|positive)\]", prompt)
        label = match.group(1) if match else "positive"
        return ProviderResult(status="ok", text=f"ANSWER: {label}",
                              raw_stdout="echo", extra={"parse_method": "echo"})


def _patch(monkeypatch, tmp_path: Path) -> None:
    """Point the workspace and report paths at the temporary project."""
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    monkeypatch.setattr(report_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")


def _authorization(cfg_path: Path, *, qualitative: bool,
                   allow_unclean: bool = False) -> dict:
    cfg = load_config(cfg_path)
    settings = qualitative_support.build_sidecar(
        registry.enabled_providers()[0], registry.enabled_providers()[0]["models"][0],
        sample_n=4, control_n=2, seed=20261116, max_calls=15, max_usd=0.1)
    return {
        "run_id": cfg["run"]["run_id"],
        "config_sha256": cfg["_meta"]["config_sha256"],
        "classification": {"max_calls": cfg["budget"]["max_calls"], "max_usd": 1.0},
        "qualitative": {"enabled": qualitative, "settings": settings,
                        "max_calls": 15, "max_usd": 0.1},
        "batch_size": 25,
        "allow_unclean_pilot": allow_unclean,
    }


def _run_to_terminal(cfg_path: Path, authorization: dict, *,
                     provider=None, qualitative: bool = False, limit: int = 80):
    status = None
    last = None
    for _ in range(limit):
        last = pipeline.advance(
            cfg_path.parent.name, cfg_path, authorization,
            classification_provider=provider,
            qualitative_provider_factory=(lambda settings: JsonStubProvider(settings))
            if qualitative else None)
        status = last.status
        if status in TERMINAL:
            break
    return status, last


def test_full_pipeline_runs_and_builds_reports(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="pipe", classes=3, n=15)
    cfg_path = study["config_path"]
    stub = StubProvider(load_config(cfg_path), label="negative")
    authorization = _authorization(cfg_path, qualitative=True)

    status, last = _run_to_terminal(cfg_path, authorization, provider=stub,
                                    qualitative=True)
    assert status == "done", last.detail
    run_dir = study["run_dir"]
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "results" / "t02_accuracy.csv").exists()
    assert (run_dir / "results" / "t11_disagreement_pairs.csv").exists()
    assert (run_dir / "pilot" / "approval.json").exists()
    assert (run_dir / "qualitative" / "explanations.csv").exists()
    assert (run_dir / "report" / "report.md").exists()
    assert (tmp_path / "reports" / "pipe" / "study_report.md").exists()
    assert (tmp_path / "reports" / "pipe" / "study_report.pdf").exists()
    predictions = pd.read_csv(run_dir / "predictions.csv", dtype="string")
    assert (predictions["status"] == "ok").all()


def test_unclean_pilot_pauses_and_then_continues(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="unclean", classes=2, n=10, max_calls=40)
    cfg_path = study["config_path"]
    # The pilot is unparseable (3 attempts x 5 rows), then the model behaves.
    provider = FlakyProvider(load_config(cfg_path), fail_first=15, label="positive")
    authorization = _authorization(cfg_path, qualitative=False)

    status, last = _run_to_terminal(cfg_path, authorization, provider=provider)
    assert status == "needs_pilot_review"
    assert last.extra.get("pilot_rows")

    authorization["allow_unclean_pilot"] = True
    status, last = _run_to_terminal(cfg_path, authorization, provider=provider)
    assert status == "done", last.detail


def test_qualitative_is_skipped_when_there_are_no_disagreements(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="agree", classes=3, n=12)
    # Rewrite the inputs so the observation carries its own label.
    names = ["negative", "neutral", "positive"]
    rows = [[f"r{i}", f"[{names[i % 3]}] observation {i} with enough words"]
            for i in range(12)]
    pd.DataFrame(rows, columns=["review_id", "review_text"]).to_csv(
        study["run_dir"] / "inputs" / "dataset.csv", index=False)
    pd.DataFrame([[f"r{i}", names[i % 3]] for i in range(12)],
                 columns=["review_id", "sentiment"]).to_csv(
        study["run_dir"] / "inputs" / "benchmark.csv", index=False)
    cfg_path = study["config_path"]
    authorization = _authorization(cfg_path, qualitative=True)

    status, last = _run_to_terminal(cfg_path, authorization,
                                    provider=EchoProvider(load_config(cfg_path)))
    assert status == "done", last.detail
    disagreements = pd.read_csv(study["run_dir"] / "results" / "t08_disagreements.csv")
    assert len(disagreements) == 0
    assert not (study["run_dir"] / "qualitative" / "explanations.csv").exists()
    assert (tmp_path / "reports" / "agree" / "study_report.md").exists()


def test_budget_stop_marks_the_run_stopped_with_pending_rows(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="budget", classes=2, n=15, max_calls=6)
    cfg_path = study["config_path"]
    stub = StubProvider(load_config(cfg_path), label="positive")
    authorization = _authorization(cfg_path, qualitative=False)

    status, last = _run_to_terminal(cfg_path, authorization, provider=stub)
    assert status == "stopped", last.detail
    frame = pd.read_csv(study["run_dir"] / "predictions.csv", dtype="string")
    assert (frame["status"] == "pending").sum() > 0


def test_a_changed_config_blocks_the_authorization(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    study = make_ui_study(tmp_path, run_id="drift", classes=2, n=8)
    cfg_path = study["config_path"]
    authorization = _authorization(cfg_path, qualitative=False)
    authorization["config_sha256"] = "0" * 64
    advance = pipeline.advance("drift", cfg_path, authorization)
    assert advance.status == "blocked"

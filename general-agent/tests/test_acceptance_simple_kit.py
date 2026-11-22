"""Acceptance: the shipped manual kit works in the three-step flow (stubs only)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from app import pipeline, propose, qualitative_support, registry, workspace
from general_agent.config import load_config
from general_agent.report import paths as report_paths
from stubs import JsonStubProvider, StubProvider

KIT = Path(__file__).resolve().parents[1] / "acceptance"


def test_kit_files_pair_up_correctly():
    observations = pd.read_csv(KIT / "observations.csv", dtype="string")
    labels = pd.read_csv(KIT / "human_labels.csv", dtype="string")
    assert len(observations) == len(labels) == 22
    assert observations["ticket_id"].is_unique
    assert labels["ref_id"].is_unique
    assert set(observations["ticket_id"]) == set(labels["ref_id"])
    assert observations["observation"].str.strip().ne("").all()


def test_kit_proposal_is_specific_and_explicit():
    observations = workspace.inspect_table(KIT / "observations.csv")
    labels = workspace.inspect_table(KIT / "human_labels.csv")
    proposal = propose.build_proposal(observations, labels)
    assert proposal.dataset_id_column == "ticket_id"
    assert proposal.text_column == "observation"
    assert proposal.benchmark_id_column == "ref_id"
    assert proposal.label_column == "human_label"
    assert {c["name"] for c in proposal.classes} == {"pos", "neu", "neg"}
    assert proposal.label_map == {"POS": "pos", "NEU": "neu", "NEG": "neg"}
    assert proposal.blocking() == []


def _kit_study(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "kit_acceptance"
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    shutil.copy2(KIT / "observations.csv", inputs / "observations.csv")
    shutil.copy2(KIT / "human_labels.csv", inputs / "human_labels.csv")

    from app.config_builder import build_config, write_config

    spec = {
        "run_id": "kit_acceptance",
        "dataset_rel": "inputs/observations.csv", "dataset_format": "csv",
        "dataset_delimiter": ",", "dataset_encoding": "utf-8",
        "dataset_id_column": "ticket_id", "text_column": "observation",
        "benchmark_rel": "inputs/human_labels.csv", "benchmark_format": "csv",
        "benchmark_delimiter": ",", "benchmark_encoding": "utf-8",
        "benchmark_id_column": "ref_id", "label_column": "human_label",
        "label_map": {"POS": "positive", "NEU": "neutral", "NEG": "negative"},
        "on_unknown_label": "error", "on_id_mismatch": "error",
        "classes": [{"name": "negative", "definition": "unfavourable"},
                    {"name": "neutral", "definition": "neither"},
                    {"name": "positive", "definition": "favourable"}],
        "ordered": False, "rows_mode": "all", "sample_n": None, "seed": 20261116,
        "task_role": "a classifier for customer feedback",
        "decision_rule": "",
        "model": {"provider": "opencode_cli", "name": "test/model"},
        "max_calls": 40, "max_usd": 1.0,
        "price_per_million": {"input": 0.0, "output": 0.0},
        "report_title": "kit acceptance", "report_author": "",
        "task_description": "", "reference_description": "",
    }
    return write_config(run_dir, build_config(spec))


def test_kit_runs_through_the_pipeline_with_stubs(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    monkeypatch.setattr(report_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")
    config_path = _kit_study(tmp_path)
    cfg = load_config(config_path)
    settings = qualitative_support.build_sidecar(
        registry.enabled_providers()[0], registry.enabled_providers()[0]["models"][0],
        sample_n=4, control_n=2, seed=20261116, max_calls=15, max_usd=0.1)
    authorization = {
        "run_id": "kit_acceptance",
        "config_sha256": cfg["_meta"]["config_sha256"],
        "classification": {"max_calls": 40, "max_usd": 1.0},
        "qualitative": {"enabled": True, "settings": settings,
                        "max_calls": 15, "max_usd": 0.1},
        "batch_size": 25,
        "allow_unclean_pilot": False,
    }
    stub = StubProvider(cfg, label="negative")
    status = None
    for _ in range(80):
        advance = pipeline.advance(
            "kit_acceptance", config_path, authorization,
            classification_provider=stub,
            qualitative_provider_factory=lambda settings: JsonStubProvider(settings))
        status = advance.status
        if status in {"done", "stopped", "error", "blocked", "needs_pilot_review"}:
            break
    assert status == "done", advance.detail

    run_dir = tmp_path / "runs" / "kit_acceptance"
    predictions = pd.read_csv(run_dir / "predictions.csv", dtype="string")
    assert len(predictions) == 22
    assert (predictions["status"] == "ok").all()
    assert (run_dir / "results" / "t02_accuracy.csv").exists()
    assert (run_dir / "qualitative" / "explanations.csv").exists()
    report = tmp_path / "reports" / "kit_acceptance" / "study_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Qualitative" in text
    assert (tmp_path / "reports" / "kit_acceptance" / "study_report.pdf").exists()

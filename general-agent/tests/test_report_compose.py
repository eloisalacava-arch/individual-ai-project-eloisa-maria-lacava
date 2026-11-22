"""The per-study report composes from tables, and says when qualitative is absent."""

from __future__ import annotations

import json

import pandas as pd

from general_agent.config import load_config
from general_agent.evaluate.compare import fmt
from general_agent.report import compose
from qualitative_helpers import evaluated_run


def _accuracy(config_path) -> str:
    cfg = load_config(config_path)
    frame = pd.read_csv(cfg["_meta"]["output_dir"] + "/results/t02_accuracy.csv")
    return fmt(frame.iloc[0]["accuracy"])


def test_study_report_has_every_section_and_matches_the_tables(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15)
    markdown, cfg = compose.compose_study(study["config_path"])

    for heading in ("## 1. Study description", "## 2. Agreement with the reference",
                    "## 3. Confusion matrix", "## 4. Per-category results",
                    "## 5. Class distribution", "## 6. Disagreement structure",
                    "## 7. Observation length (exploratory)",
                    "## 8. Qualitative findings", "## 9. Interpretation",
                    "## 10. Conclusions", "## 11. Limitations",
                    "## 12. Reproducibility"):
        assert heading in markdown, heading

    assert _accuracy(study["config_path"]) in markdown
    assert cfg["run"]["run_id"] in markdown
    assert "Source: `results/t05_per_class_metrics.csv`" in markdown


def test_study_report_states_when_qualitative_has_not_run(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "Qualitative analysis has not been run" in markdown
    assert "mixed_or_conflicting_signals" not in markdown  # no invented reason
    assert "AI-generated hypotheses" not in markdown


def test_write_study_report_creates_markdown_and_metadata(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    out = tmp_path / "reports" / "ui_study"
    path = compose.write_study_report(study["config_path"], out_dir=out)
    assert path.exists()
    meta = json.loads((out / "report_meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "ui_study"
    assert meta["qualitative"] is False
    assert any(name.startswith("results/t0") for name in meta["sources"])


def test_resolve_study_config_finds_the_run_directory(tmp_path, monkeypatch):
    from general_agent.report import paths
    study = evaluated_run(tmp_path, classes=2, n=8)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    resolved = compose.resolve_study_config(study["run_dir"].name)
    assert resolved is not None
    assert resolved.name == "study.yaml"
    assert resolved.parent == study["run_dir"]

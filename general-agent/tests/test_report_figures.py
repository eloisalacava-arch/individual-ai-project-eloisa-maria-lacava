"""Figures are regenerated from the verified tables as PNG."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from general_agent.config import load_config
from general_agent.evaluate.compare import run_comparison
from general_agent.report import figures
from qualitative_helpers import evaluated_run

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_study_figures_are_png(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15)
    cfg = load_config(study["config_path"])
    for builder in (figures.confusion_matrix_png, figures.per_class_recall_png):
        data = builder(cfg)
        assert data.startswith(PNG_MAGIC)
        assert len(data) > 500


def test_comparison_figures_are_png(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15)
    out = tmp_path / "comparison"
    assert run_comparison([study["config_path"]], output_dir=out) == 0
    for builder in (figures.comparison_metrics_png,
                    figures.comparison_class_recall_png,
                    figures.comparison_severity_png):
        data = builder(out)
        assert data.startswith(PNG_MAGIC)
        assert len(data) > 500


def test_missing_table_is_an_error_not_a_silent_figure(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    cfg = load_config(study["config_path"])
    (study["run_dir"] / "results" / "t05_per_class_metrics.csv").unlink()
    with pytest.raises(FileNotFoundError):
        figures.per_class_recall_png(cfg)

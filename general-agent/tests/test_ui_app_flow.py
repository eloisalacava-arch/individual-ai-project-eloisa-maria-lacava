"""Drive the interface end to end through Streamlit's headless harness.

Uploads are injected as session state because the harness cannot populate a
file uploader; everything after that (spec assembly, materialising inputs,
writing the config, configure, audit) runs the real view code. No model call is
made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import workspace  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_validate_step_creates_a_valid_study(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")

    dataset = workspace.stage_upload(
        "s", "dataset", "reviews.csv",
        b"review_id,review_text\nr1,hello world this is a review\n"
        b"r2,another observation with words\n")
    benchmark = workspace.stage_upload(
        "s", "benchmark", "labels.csv",
        b"review_id,sentiment\nr1,positive\nr2,negative\n")
    dataset_meta = workspace.inspect_table(dataset["path"])
    benchmark_meta = workspace.inspect_table(benchmark["path"])

    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["flow"] = "advanced"
    state = {
        "step": 5,
        "study_name": "ui-integration",
        "dataset_upload": dataset,
        "benchmark_upload": benchmark,
        "dataset_meta": dataset_meta,
        "benchmark_meta": benchmark_meta,
        "dataset_id_column": "review_id",
        "text_column": "review_text",
        "benchmark_id_column": "review_id",
        "label_column": "sentiment",
        "classes": [
            {"name": "positive", "definition": "favourable"},
            {"name": "negative", "definition": "unfavourable"},
        ],
        "label_map": {"positive": "positive", "negative": "negative"},
        "on_unknown_label": "error",
        "on_id_mismatch": "error",
        "rows_mode": "all",
        "seed": 20261117,
        "max_calls": 20,
        "max_usd": 1.0,
        "consent": True,
    }
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception

    next(button for button in app.button
         if button.label == "Create study and validate").click()
    app.run()
    assert not app.exception

    run_dir = tmp_path / "runs" / "ui-integration"
    assert (run_dir / "study.yaml").exists()
    decisions = workspace.read_decisions("ui-integration")
    assert decisions["status"] == "resolved"
    manifest = workspace.load_manifest("ui-integration")
    assert manifest["checks"]["status"] == "PASS"
    assert manifest["row_counts"]["dataset_analysed"] == 2


def test_sealing_closes_the_column_and_category_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    run_dir = tmp_path / "runs" / "sealed_study"
    (run_dir / "blind").mkdir(parents=True)
    (run_dir / "sealed").mkdir(parents=True)
    (run_dir / "blind" / "blind.csv").write_text("id,text\nr1,hello\n", encoding="utf-8")
    (run_dir / "sealed" / "benchmark.csv").write_text(
        "id,sentiment\nr1,positive\n", encoding="utf-8")
    # The seal marker lives at the run root and stays readable while the engine
    # locks the sealed directory during classification.
    (run_dir / "blind_public_info.json").write_text("{}", encoding="utf-8")
    assert workspace.stage_flags("sealed_study")["sealed"] is True

    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["flow"] = "advanced"
    app.session_state["run_id"] = "sealed_study"
    app.session_state["step"] = 2
    app.run()
    assert not app.exception
    texts = " ".join(element.value for element in app.markdown)
    assert "sealed" in texts.lower()
    disabled = {button.label: button.disabled for button in app.button}
    assert next(v for k, v in disabled.items() if k.startswith("2.")) is True
    assert next(v for k, v in disabled.items() if k.startswith("3.")) is True

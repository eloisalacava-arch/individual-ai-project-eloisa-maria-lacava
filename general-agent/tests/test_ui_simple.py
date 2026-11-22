"""P5: the simple three-step flow end to end, with a stub provider.

No live model calls: ``app.engine_adapter.run`` is wrapped to inject a
StubProvider, exactly as the Analyze screen would build the real one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import engine_adapter, workspace  # noqa: E402
from general_agent.config import load_config  # noqa: E402
from general_agent.report import paths as report_paths  # noqa: E402
from stubs import StubProvider  # noqa: E402
from test_acceptance_new_user import _benchmark_bytes, _dataset_bytes  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_three_step_flow_completes_with_a_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    monkeypatch.setattr(report_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")

    original_run = engine_adapter.run

    def stub_run(config_path, mode="full", limit=None, resume=False, provider=None):
        if provider is None:
            classes = load_config(config_path)["labels"]["classes"]
            provider = StubProvider(load_config(config_path),
                                    label=classes[0]["name"])
        return original_run(config_path, mode=mode, limit=limit, resume=resume,
                            provider=provider)

    monkeypatch.setattr(engine_adapter, "run", stub_run)

    dataset = workspace.stage_upload("s", "dataset", "observations.csv",
                                     _dataset_bytes())
    benchmark = workspace.stage_upload("s", "benchmark", "human_labels.csv",
                                       _benchmark_bytes())
    dataset_meta = workspace.inspect_table(dataset["path"])
    benchmark_meta = workspace.inspect_table(benchmark["path"])

    app = AppTest.from_file(str(APP), default_timeout=120)
    for key, value in (("dataset_upload", dataset), ("benchmark_upload", benchmark),
                       ("dataset_meta", dataset_meta),
                       ("benchmark_meta", benchmark_meta)):
        app.session_state[key] = value
    app.run()
    assert not app.exception and not app.error
    # Simple mode is the default.
    assert any("1 · Upload" in header.value for header in app.subheader)

    next(field for field in app.text_input
         if field.key == "__w_study_name").set_value("simple_acceptance")
    app.run()
    _button(app, "Continue to review").click()
    app.run()
    assert not app.exception
    assert any("2 · Review" in header.value for header in app.subheader)
    # The proposed definitions are placeholders, so the review blocks with an
    # explicit error until they are replaced.
    assert any("placeholder" in element.value.lower() for element in app.error)

    # The proposal filled the columns, categories, and mapping.
    assert app.session_state["dataset_id_column"] == "doc_id"
    assert app.session_state["text_column"] == "comment"
    assert app.session_state["benchmark_id_column"] == "record_id"
    assert app.session_state["label_column"] == "verdict"
    # The synthetic benchmark labels are 0/1, so polarity is never inferred:
    # generic category names are proposed with an explicit mapping.
    names = [c["name"] for c in app.session_state["classes"]]
    assert len(names) == 2 and all(name.startswith("category-") for name in names)
    mapping = app.session_state["label_map"]
    assert set(mapping) == {"0", "1"}
    assert set(mapping.values()) == set(names)
    # The proposed definitions are placeholders, so the review must block until
    # they are replaced with what each category means.
    blocking = app.session_state["proposal"]["blocking"]
    assert any("placeholder" in problem.lower() for problem in blocking)

    app.session_state["classes_editor"] = {
        "edited_rows": {index: {"definition": f"meaningful definition for {name}"}
                        for index, name in enumerate(names)},
        "added_rows": [],
        "deleted_rows": [],
    }
    app.run()
    assert not app.error
    assert app.session_state["proposal"]["blocking"] == []

    # Keep the test offline: turn off the qualitative stage.
    next(box for box in app.checkbox
         if box.key == "__w_qualitative_enabled").uncheck()
    app.run()
    _button(app, "Continue to analyze").click()
    app.run()
    assert any("3 · Analyze & Download" in header.value for header in app.subheader)

    # The budget separates classification from the optional qualitative part and
    # shows the combined maximum before authorization.
    budget = " ".join(element.value for element in app.markdown
                      if "Combined maximum" in element.value)
    assert "Classification (five-row pilot" in budget
    assert "Combined maximum" in budget
    assert "Optional qualitative explanations" in budget

    next(box for box in app.checkbox if box.key == "__w_authorize_ack").check()
    app.run()
    _button(app, "Authorize and run").click()

    # The pipeline drives itself; give the harness a few runs to finish.
    for _ in range(40):
        app.run()
        if any("complete" in message.value.lower() for message in app.success):
            break
    assert not app.exception
    assert not app.error

    run_dir = tmp_path / "runs" / "simple_acceptance"
    assert run_dir.exists()
    cfg = load_config(run_dir / "study.yaml")
    assert cfg["run"]["run_id"] == "simple_acceptance"
    assert (run_dir / "results" / "t02_accuracy.csv").exists()
    assert (run_dir / "pilot" / "approval.json").exists()
    # Qualitative was disabled, so no explanations were produced.
    assert not (run_dir / "qualitative" / "explanations.csv").exists()
    report = tmp_path / "reports" / "simple_acceptance" / "study_report.md"
    assert report.exists()
    assert (tmp_path / "reports" / "simple_acceptance" / "study_report.pdf").exists()

    downloads = [item.label for item in app.get("download_button")]
    assert any("Download Full Report (PDF)" in label for label in downloads)

    # The new study is unrelated to the shipped case studies.
    predictions = pd.read_csv(run_dir / "predictions.csv", dtype="string")
    assert all(identifier.startswith("AC-") for identifier in predictions["id"])


def test_advanced_mode_toggle_shows_the_seven_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert any("1 · Upload" in header.value for header in app.subheader)
    _button(app, "Advanced mode").click()
    app.run()
    assert not app.exception
    assert any("1 · Data" in header.value for header in app.subheader)
    labels = [button.label for button in app.button]
    assert any(label.startswith("7.") for label in labels)
    if any(label == "Switch to simple mode" for label in labels):
        _button(app, "Switch to simple mode").click()
        app.run()
        assert any("1 · Upload" in header.value for header in app.subheader)

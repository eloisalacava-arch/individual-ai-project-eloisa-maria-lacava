"""Headless smoke tests for the Streamlit app.

These render the real script with Streamlit's testing harness. They do not drive
file uploads (not supported by the harness); upload handling is covered in
``test_ui_workspace.py`` and the stage flow in ``test_ui_engine_adapter.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _render(**state):
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["flow"] = "advanced"
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    return app


def test_app_starts_on_step_one_without_exception():
    app = _render()
    assert not app.exception
    assert any("1 · Data" in header.value for header in app.subheader)


def test_steps_are_gated_before_the_data_is_uploaded():
    app = _render()
    buttons = {button.label: button.disabled for button in app.button}
    assert buttons["Continue to columns"] is True
    assert buttons["2. Columns"] is True
    assert buttons["6. Classify"] is True
    assert buttons["7. Results"] is True


def test_examples_page_renders_read_only():
    app = _render(page="examples")
    assert not app.exception
    assert any("Case studies" in header.value for header in app.subheader)


def test_results_page_reads_a_shipped_case_study():
    from app import workspace
    run = workspace.run_dir("imdb_case_study")
    if not (run / "results" / "t02_accuracy.csv").exists():
        pytest.skip("the IMDb case study is not present in this checkout")
    if not workspace.stage_flags("imdb_case_study").get("sealed"):
        pytest.skip("the full IMDb run (including blind text) is not present in this checkout")
    app = _render(step=7, run_id="imdb_case_study")
    assert not app.exception
    assert not app.error  # a command-line study must not raise a hidden error
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Rows analysed"] == "200"
    assert metrics["Accuracy"] == "0.970"

"""Streamlit entry point for the general-agent interface.

Run from the ``general-agent`` directory:

    uv run streamlit run app/streamlit_app.py

Two flows share one engine: a simple three-step flow (Upload → Review →
Analyze & Download) for first-time users, and the original seven-step flow behind
"Advanced mode".
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st  # noqa: E402

from app import state, theme, workspace  # noqa: E402
from app.views import (  # noqa: E402
    analyze,
    examples,
    review,
    step_categories,
    step_classify,
    step_columns,
    step_data,
    step_model,
    step_results,
    step_validate,
    upload,
)

ADVANCED_VIEWS = {
    1: step_data.render,
    2: step_columns.render,
    3: step_categories.render,
    4: step_model.render,
    5: step_validate.render,
    6: step_classify.render,
    7: step_results.render,
}
SIMPLE_VIEWS = {
    1: upload.render,
    2: review.render,
    3: analyze.render,
}


def main() -> None:
    st.set_page_config(
        page_title="AI classification and evaluation",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(theme.CUSTOM_CSS, unsafe_allow_html=True)
    state.init_state()
    _sidebar()

    if st.session_state["page"] == "examples":
        examples.render()
        return

    flow = st.session_state.get("flow", "simple")
    st.title("AI classification against a human benchmark")
    if flow == "advanced":
        st.markdown(
            "<div class='ga-subtitle'>Advanced mode: every stage, one screen at a "
            "time.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='ga-subtitle'>Upload two files, check what was proposed, "
            "then authorize one bounded run.</div>", unsafe_allow_html=True)
    st.write("")

    views = ADVANCED_VIEWS if flow == "advanced" else SIMPLE_VIEWS
    default = step_data.render if flow == "advanced" else upload.render
    render = views.get(int(st.session_state["step"]), default)
    try:
        render()
    except Exception as exc:  # keep the interface usable; offer the details
        st.error(f"Something went wrong in this step: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc(), language=None)


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### Classification studio")
        if st.button("Start a new study", width="stretch"):
            state.reset_study()
            st.rerun()

        st.markdown("---")
        flow = st.session_state.get("flow", "simple")
        labels = state.STEP_LABELS if flow == "advanced" else state.SIMPLE_LABELS
        flags = state.run_flags()
        for number, label in labels:
            allowed = state.step_allowed(number)
            current = int(st.session_state["step"]) == number and \
                st.session_state["page"] == "study"
            if st.button(
                f"{number}. {label}" + ("  ←" if current else ""),
                key=f"nav_{flow}_{number}",
                disabled=not allowed,
                width="stretch",
            ):
                state.go(number)
        if flags:
            st.markdown(
                theme.chip(f"sealed {flags.get('sealed')}")
                + theme.chip(f"classified {flags.get('classified')}")
                + theme.chip(f"evaluated {flags.get('evaluated')}"),
                unsafe_allow_html=True)

        if flow == "advanced":
            if st.button("Switch to simple mode", width="stretch"):
                st.session_state["flow"] = "simple"
                st.session_state["step"] = 1
                st.rerun()
        else:
            if st.button("Advanced mode", width="stretch"):
                st.session_state["flow"] = "advanced"
                st.session_state["step"] = 1
                st.rerun()

        runs = workspace.list_runs()
        if runs:
            st.markdown("---")
            st.markdown("#### Previous runs")
            run_ids = [run["run_id"] for run in runs]
            choice = st.selectbox(
                "Open a run", run_ids, key="open_run_select",
                format_func=lambda value: next(
                    run["title"] for run in runs if run["run_id"] == value))
            if st.button("Open", width="stretch"):
                state.open_run(choice)

        st.markdown("---")
        if st.button("Case studies", width="stretch"):
            state.go_examples()

        with st.expander("Method notes"):
            st.markdown(
                "- The classifier sees only the id and the text of each row.\n"
                "- Human labels are sealed before classification and opened only "
                "after predictions are frozen.\n"
                "- One authorization covers the whole bounded run; the pilot is "
                "approved automatically only when it parses cleanly.\n"
                "- Sampling is seeded and never stratified.\n"
                "- No causal claims: results are associations on one sample.\n"
                "- Credentials stay in the environment; this interface never "
                "stores or shows a key.")


if __name__ == "__main__":
    main()

"""Step 1 (advanced) - upload the observations and the human benchmark."""

from __future__ import annotations

import streamlit as st

from app import workspace
from app.state import go, p_text
from app.uploads import render_uploaders


def render() -> None:
    st.subheader("1 · Data")
    st.caption("Upload the observations to classify and a separate file of human "
               "labels. The AI will never see the label file.")

    run_id = st.session_state.get("run_id")
    flags = workspace.stage_flags(run_id) if run_id else {}
    locked = bool(flags.get("sealed"))

    # A persistent widget: a plain widget key would be cleared when this step is
    # left, and the study name would be lost before the run is created.
    p_text("Study name", "study_name", placeholder="e.g. product reviews pilot")

    render_uploaders(locked=locked)

    if locked:
        st.warning("This run is already sealed. Start a new study to use different "
                   "data.")

    ready = bool(st.session_state.get("dataset_meta")) and bool(
        st.session_state.get("benchmark_meta"))
    if st.button("Continue to columns", type="primary", disabled=not ready or locked):
        go(2)
    if not ready:
        st.caption("Upload both files to continue.")

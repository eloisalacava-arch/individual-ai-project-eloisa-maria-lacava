"""Shared upload handling for the simple and advanced flows."""

from __future__ import annotations

import hashlib

import streamlit as st

from app import workspace


def reset_downstream() -> None:
    st.session_state.update({
        "dataset_id_column": None, "text_column": None,
        "benchmark_id_column": None, "label_column": None,
        "label_values": [], "label_map": {}, "label_excluded": [],
        "run_id": None, "last_stage": None, "proposal": None,
        "proposal_key": None, "authorization": None, "stop_requested": False,
    })


def handle_upload(role: str, uploaded, state_key: str, meta_key: str) -> None:
    if uploaded is None:
        return
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    existing = st.session_state.get(state_key)
    if existing and existing.get("sha256") == digest:
        return
    staged = workspace.stage_upload(st.session_state["session_id"], role,
                                    uploaded.name, data)
    try:
        meta = workspace.inspect_table(staged["path"])
    except Exception as exc:  # a malformed upload must not crash the app
        st.error(f"Could not read {uploaded.name}: {exc}")
        return
    st.session_state[state_key] = staged
    st.session_state[meta_key] = meta
    reset_downstream()


def _summary(meta_key: str) -> None:
    meta = st.session_state.get(meta_key)
    if not meta:
        return
    staged = st.session_state.get("dataset_upload" if meta_key == "dataset_meta"
                                  else "benchmark_upload") or {}
    st.markdown(
        f"<div class='ga-card'><b>{staged.get('original_name', 'file')}</b><br>"
        f"<span class='ga-muted'>{meta['n_rows']} rows · "
        f"{len(meta['columns'])} columns · {meta['format']}"
        f"{' · delimiter ' + repr(meta['delimiter']) if meta['format'] == 'csv' else ''}"
        f" · {meta['encoding']}</span></div>",
        unsafe_allow_html=True)
    with st.expander("Preview first rows"):
        st.dataframe(meta["preview"], width="stretch", hide_index=True)


def render_uploaders(locked: bool = False) -> None:
    """The two upload columns and their summaries."""
    left, right = st.columns(2)
    with left:
        st.markdown("**Dataset to classify**")
        st.caption("An ID column and a text column. Extra columns are allowed and "
                   "never sent to the model.")
        uploaded = st.file_uploader(
            "Observations", type=["csv", "tsv", "txt", "jsonl", "ndjson"],
            key="dataset_uploader", disabled=locked,
            label_visibility="collapsed")
        handle_upload("dataset", uploaded, "dataset_upload", "dataset_meta")
        _summary("dataset_meta")
    with right:
        st.markdown("**Human benchmark**")
        st.caption("The same IDs and the human label for each observation. This "
                   "file is sealed before classification.")
        uploaded = st.file_uploader(
            "Benchmark", type=["csv", "tsv", "txt", "jsonl", "ndjson"],
            key="benchmark_uploader", disabled=locked,
            label_visibility="collapsed")
        handle_upload("benchmark", uploaded, "benchmark_upload", "benchmark_meta")
        _summary("benchmark_meta")

"""Step 2 - choose the ID, text, and human-label columns."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from general_agent.io import normalize_ids
from app import workspace
from app.state import go, p_check, p_select

ID_HINT = re.compile(r"(^id$|_id$|^uid$|^key$|row_?id|doc_?id|record_?id|_key$)")
TEXT_HINT = re.compile(r"(text|body|review|sentence|comment|content|description|"
                       r"message|response|article|headline|title|abstract)")
LABEL_HINT = re.compile(r"(label|sentiment|class|category|target|annotation|gold|"
                        r"truth|topic|rating|polarity|stance|outcome)")


def _guess(columns: list[str], pattern: re.Pattern) -> str | None:
    for column in columns:
        if pattern.search(column.lower()):
            return column
    return None


def _match_summary(dataset_meta: dict, benchmark_meta: dict,
                   dataset_id: str, benchmark_id: str) -> dict | None:
    if not (dataset_id and benchmark_id):
        return None
    if dataset_id not in dataset_meta["frame"].columns:
        return None
    if benchmark_id not in benchmark_meta["frame"].columns:
        return None
    left = set(normalize_ids(dataset_meta["frame"][dataset_id]).dropna())
    right = set(normalize_ids(benchmark_meta["frame"][benchmark_id]).dropna())
    return {
        "dataset_ids": len(left),
        "benchmark_ids": len(right),
        "matched": len(left & right),
        "only_dataset": len(left - right),
        "only_benchmark": len(right - left),
    }


def render() -> None:
    st.subheader("2 · Columns")
    run_id = st.session_state.get("run_id")
    if run_id and workspace.stage_flags(run_id).get("sealed"):
        st.info("This study is sealed. The human labels are hidden until "
                "evaluation; start a new study to change the columns.")
        return
    st.caption("Only the ID and the text are sent to the AI. The human label is "
               "used only after classification.")

    dataset_meta = st.session_state["dataset_meta"]
    benchmark_meta = st.session_state["benchmark_meta"]
    dataset_columns = dataset_meta["columns"]
    benchmark_columns = benchmark_meta["columns"]

    p_check("The dataset has no ID column: create one from row numbers", "make_id")
    make_id = st.session_state["make_id"]
    if make_id:
        st.caption("A column `observation_id` will be added when the study is "
                   "created. No rows are dropped.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Dataset**")
        if make_id:
            st.text_input("ID column", value="observation_id (generated)",
                          disabled=True)
        else:
            if st.session_state.get("dataset_id_column") not in dataset_columns:
                guess = _guess(dataset_columns, ID_HINT)
                if guess:
                    st.session_state["dataset_id_column"] = guess
            p_select("ID column", dataset_columns, "dataset_id_column")
        if st.session_state.get("text_column") not in dataset_columns:
            guess = _guess(dataset_columns, TEXT_HINT)
            if guess:
                st.session_state["text_column"] = guess
        p_select("Text column", dataset_columns, "text_column")
        st.dataframe(_truncate(dataset_meta["preview"]), width="stretch",
                     hide_index=True)
    with right:
        st.markdown("**Human benchmark**")
        if st.session_state.get("benchmark_id_column") not in benchmark_columns:
            guess = _guess(benchmark_columns, ID_HINT)
            if guess:
                st.session_state["benchmark_id_column"] = guess
        p_select("ID column", benchmark_columns, "benchmark_id_column")
        if st.session_state.get("label_column") not in benchmark_columns:
            guess = _guess(benchmark_columns, LABEL_HINT)
            if guess:
                st.session_state["label_column"] = guess
        p_select("Human label column", benchmark_columns, "label_column")
        st.dataframe(_truncate(benchmark_meta["preview"]), width="stretch",
                     hide_index=True)

    dataset_id = None if make_id else st.session_state.get("dataset_id_column")
    summary = _match_summary(dataset_meta, benchmark_meta, dataset_id or "",
                             st.session_state.get("benchmark_id_column") or "")
    if summary:
        if summary["only_dataset"] or summary["only_benchmark"]:
            st.warning(
                f"IDs do not match exactly: {summary['only_dataset']} dataset ID(s) "
                f"have no benchmark row and {summary['only_benchmark']} benchmark "
                f"ID(s) have no dataset row. You can keep only matching IDs in the "
                "next step.")
        else:
            st.success(f"{summary['matched']} IDs match in both files.")

    ready = bool(st.session_state.get("text_column")) and bool(
        st.session_state.get("benchmark_id_column")) and bool(
        st.session_state.get("label_column"))
    if st.button("Continue to categories", type="primary", disabled=not ready):
        go(3)


def _truncate(frame: pd.DataFrame, width: int = 70) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        out[column] = out[column].astype("string").str.replace(
            r"\s+", " ", regex=True).str.slice(0, width)
    return out

"""Step 3 - define the categories and map the human labels onto them."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from app import workspace
from app.state import go, p_check, p_radio

UNMAPPED = "(unmapped)"
EXCLUDE = "(exclude rows with this label)"


def _class_names() -> list[str]:
    return [entry["name"].strip() for entry in st.session_state["classes"]
            if entry.get("name", "").strip()]


def _label_counts() -> pd.Series:
    meta = st.session_state.get("benchmark_meta")
    label_column = st.session_state.get("label_column")
    if not meta or not label_column or label_column not in meta["frame"].columns:
        return pd.Series(dtype=int)
    values = meta["frame"][label_column].astype("string").str.strip()
    values = values[values.notna() & (values != "")]
    return values.value_counts()


def render() -> None:
    st.subheader("3 · Categories")
    run_id = st.session_state.get("run_id")
    if run_id and workspace.stage_flags(run_id).get("sealed"):
        st.info("This study is sealed. The human labels are hidden until "
                "evaluation; start a new study to change the categories.")
        return
    st.caption("The AI receives only these names and definitions, never the human "
               "label values.")

    editor_key = "classes_editor"
    current = pd.DataFrame(st.session_state["classes"] or [], columns=["name", "definition"])
    edited = st.data_editor(
        current,
        num_rows="dynamic",
        key=editor_key,
        hide_index=True,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn(
                "Category", required=True,
                help="One slug: letters, digits, dot, underscore or hyphen."),
            "definition": st.column_config.TextColumn(
                "What it means", required=True, width="large",
                help="Shown to the AI in the prompt."),
        },
    )
    rows = []
    for record in edited.to_dict(orient="records"):
        name = str(record.get("name") or "").strip()
        definition = str(record.get("definition") or "").strip()
        if name or definition:
            rows.append({"name": name, "definition": definition})
    st.session_state["classes"] = rows

    p_check(
        "These categories have a natural order",
        "ordered",
        help="Enables quadratic weighted kappa in addition to unweighted kappa. "
             "List them in the intended order, lowest first.")
    if st.session_state["ordered"] and len(_class_names()) < 3:
        st.caption("With two categories, weighted and unweighted kappa are the "
                   "same; the order has no effect.")

    classes = _class_names()
    counts = _label_counts()
    st.markdown("#### Map the human labels")
    if counts.empty:
        st.info("Choose the human label column in the previous step to map values.")
    else:
        st.caption("The AI is never told these values. Map each one to a category, "
                   "or mark it to be excluded.")
        mapping: dict[str, str] = dict(st.session_state.get("label_map") or {})
        excluded: list[str] = list(st.session_state.get("label_excluded") or [])
        for value, count in counts.items():
            key = "map_" + hashlib.md5(str(value).encode("utf-8")).hexdigest()[:10]
            options = [UNMAPPED] + classes + [EXCLUDE]
            if value in mapping and mapping[value] in classes:
                default = mapping[value]
            elif value in excluded:
                default = EXCLUDE
            else:
                default = UNMAPPED
            choice = st.selectbox(
                f"{value!r}  ({int(count)} rows)", options, key=key,
                index=options.index(default))
            if choice in classes:
                mapping[value] = choice
                if value in excluded:
                    excluded.remove(value)
            elif choice == EXCLUDE:
                mapping.pop(value, None)
                if value not in excluded:
                    excluded.append(value)
            else:
                mapping.pop(value, None)
                if value in excluded:
                    excluded.remove(value)
        st.session_state["label_map"] = mapping
        st.session_state["label_excluded"] = excluded
        st.session_state["on_unknown_label"] = "exclude" if excluded else "error"
        if excluded:
            st.warning(f"{len(excluded)} label value(s) will be excluded, with "
                       f"counts printed: {excluded}")
        unmapped = [value for value in counts.index
                    if value not in mapping and value not in excluded]
        if unmapped:
            st.error(f"{len(unmapped)} label value(s) are unmapped: {unmapped}")

    st.markdown("#### If IDs do not match")
    p_radio(
        "Mismatched IDs",
        options=["error", "inner_join"],
        key="on_id_mismatch",
        format_func=lambda value: {
            "error": "Stop and let me fix the files",
            "inner_join": "Keep only IDs present in both files (counts printed)",
        }[value],
        help="The engine prints rows before and after for every drop.")

    problems = _problems()
    for problem in problems:
        st.warning(problem)
    if st.button("Continue to model and limits", type="primary",
                 disabled=bool(problems)):
        go(4)


def _problems() -> list[str]:
    problems = []
    classes = st.session_state["classes"]
    valid = [c for c in classes if c.get("name") and c.get("definition")]
    if len(valid) < 2:
        problems.append("Define at least two categories with a name and a definition.")
    counts = _label_counts()
    if not counts.empty:
        mapping = st.session_state.get("label_map", {})
        excluded = set(counts.index) - set(mapping)
        if st.session_state.get("on_unknown_label") == "error":
            unmapped = [value for value in counts.index if value not in mapping]
            if unmapped:
                problems.append(
                    f"Map every human label value, or choose to exclude: {unmapped}")
        present = set(mapping.values())
        declared_missing = [name for name in _class_names() if name not in present]
        if declared_missing and not excluded:
            problems.append(
                f"These categories never appear in the mapped labels: "
                f"{declared_missing}. Add data, remove the category, or map a value "
                "to it.")
    return problems

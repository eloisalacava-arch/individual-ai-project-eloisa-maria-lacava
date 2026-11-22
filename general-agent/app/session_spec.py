"""Build a study config specification from the current session choices.

Shared by the advanced Validate step and the simple Analyze pipeline so the two
flows cannot drift. Reads ``st.session_state`` and the materialised input paths.
"""

from __future__ import annotations

import streamlit as st

from app import registry


def build_spec(run_id: str, materialised: dict) -> dict:
    provider = registry.find_provider(st.session_state.get("provider_id"))
    model = registry.find_model(st.session_state.get("provider_id"),
                                st.session_state.get("model_name"))
    if provider is None or model is None:
        raise ValueError("Choose a provider and model before creating the study.")
    classes = [c for c in st.session_state.get("classes", [])
               if c.get("name", "").strip() and c.get("definition", "").strip()]
    benchmark_meta = st.session_state["benchmark_meta"]
    class_names = [entry["name"] for entry in classes]
    return {
        "run_id": run_id,
        "dataset_rel": materialised["dataset_rel"],
        "dataset_format": materialised["dataset_format"],
        "dataset_delimiter": materialised["dataset_delimiter"],
        "dataset_encoding": materialised["dataset_encoding"],
        "dataset_id_column": materialised["dataset_id_column"]
        or st.session_state.get("dataset_id_column"),
        "text_column": st.session_state.get("text_column"),
        "benchmark_rel": materialised["benchmark_rel"],
        "benchmark_format": benchmark_meta["format"],
        "benchmark_delimiter": benchmark_meta["delimiter"],
        "benchmark_encoding": benchmark_meta["encoding"],
        "benchmark_id_column": st.session_state.get("benchmark_id_column"),
        "label_column": st.session_state.get("label_column"),
        "label_map": {k: v for k, v in (st.session_state.get("label_map") or {}).items()
                      if v},
        "on_unknown_label": st.session_state.get("on_unknown_label", "error"),
        "on_id_mismatch": st.session_state.get("on_id_mismatch", "error"),
        "classes": classes,
        "ordered": bool(st.session_state.get("ordered")),
        "rows_mode": st.session_state.get("rows_mode", "all"),
        "sample_n": st.session_state.get("sample_n"),
        "seed": st.session_state.get("seed", 20261116),
        "task_role": st.session_state.get("task_role")
        or "a classifier for short observations",
        "decision_rule": st.session_state.get("decision_rule", ""),
        "model": registry.model_block(provider, model),
        "max_calls": st.session_state.get("max_calls"),
        "max_usd": st.session_state.get("max_usd"),
        "price_per_million": registry.price_block(model),
        "report_title": (st.session_state.get("report_title")
                         or st.session_state.get("study_name") or run_id),
        "report_author": st.session_state.get("report_author", ""),
        "task_description": (st.session_state.get("task_description")
                             or f"{len(class_names)}-category classification of the "
                                "uploaded observations."),
        "reference_description": (
            st.session_state.get("reference_description")
            or "Human labels from "
               f"`{(st.session_state.get('benchmark_upload') or {}).get('original_name', 'the benchmark file')}`."),
    }

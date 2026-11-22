"""Session state, navigation, and shared lookups.

Streamlit re-runs the whole script on every interaction. Study state lives in
``st.session_state`` for the draft, and on disk once a run workspace exists, so
a refresh never loses a sealed or classified run.
"""

from __future__ import annotations

import uuid

import streamlit as st

from app import workspace

STEP_LABELS = [
    (1, "Data"), (2, "Columns"), (3, "Categories"),
    (4, "Model and limits"), (5, "Validate and seal"),
    (6, "Classify"), (7, "Results"),
]

# The default, three-step flow.
SIMPLE_LABELS = [
    (1, "Upload"), (2, "Review"), (3, "Analyze & Download"),
]

DEFAULTS = {
    "page": "study",
    "flow": "simple",          # simple | advanced
    "step": 1,
    "session_id": None,
    "stop_requested": False,
    "auto_run": True,
    "authorization": None,
    "proposal": None,
    "proposal_key": None,
    "pilot_override": False,
    "study_name": "",
    "dataset_upload": None,
    "benchmark_upload": None,
    "dataset_meta": None,
    "benchmark_meta": None,
    "dataset_id_column": None,
    "text_column": None,
    "benchmark_id_column": None,
    "label_column": None,
    "make_id": False,
    "classes": [
        {"name": "positive", "definition": ""},
        {"name": "negative", "definition": ""},
    ],
    "ordered": False,
    "label_values": [],
    "label_map": {},
    "label_excluded": [],
    "on_unknown_label": "error",
    "on_id_mismatch": "error",
    "provider_id": None,
    "model_name": None,
    "rows_mode": "all",
    "sample_n": 100,
    "seed": 20261116,
    "max_calls": 30,
    "max_usd": 0.10,
    "consent": False,
    "task_role": "a classifier for short observations",
    "decision_rule": "",
    "report_title": "",
    "report_author": "",
    "qualitative_enabled": True,
    "qualitative_sample_n": 15,
    "qualitative_control_n": 5,
    "qualitative_max_calls": 30,
    "qualitative_max_usd": 0.25,
    "proposal_stats": None,
    "run_id": None,
    "last_stage": None,
    "pilot_note": "",
    "batch_size": 25,
}

STUDY_KEYS = list(DEFAULTS.keys())


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    if not st.session_state["session_id"]:
        st.session_state["session_id"] = uuid.uuid4().hex[:12]
    if st.session_state.get("provider_id") is None:
        from app import registry
        providers = registry.enabled_providers()
        if providers:
            st.session_state["provider_id"] = providers[0]["id"]
            models = providers[0].get("models", [])
            if models:
                st.session_state["model_name"] = models[0]["name"]
    if not st.session_state.get("report_title"):
        st.session_state["report_title"] = "AI classification against a human benchmark"


def reset_study() -> None:
    preserved = {"session_id": st.session_state.get("session_id"),
                 "flow": st.session_state.get("flow", "simple")}
    for key in list(st.session_state):
        if key in STUDY_KEYS or key.startswith(
                ("classes_editor", "map_", "cls_", "qu_")):
            st.session_state.pop(key, None)
    init_state()
    st.session_state.update(preserved)
    st.session_state["step"] = 1
    st.session_state["page"] = "study"


# --------------------------------------------------------------------------- #
# Persistent widgets                                                          #
#                                                                            #
# Streamlit deletes a widget's session key when the widget stops being        #
# rendered, so a widget-backed key cannot carry a value from one step to the  #
# next. These helpers render under a private widget key and mirror the value  #
# into the plain ``key`` that later steps read.                               #
# --------------------------------------------------------------------------- #

def p_select(label, options, key, **kwargs):
    options = list(options)
    durable = st.session_state.get(key)
    if durable not in options:
        st.session_state.pop(key, None)
        durable = None
    index = options.index(durable) if durable in options else int(kwargs.pop("index", 0))
    value = st.selectbox(label, options, index=min(index, len(options) - 1),
                         key=f"__w_{key}", **kwargs)
    st.session_state[key] = value
    return value


def p_radio(label, options, key, **kwargs):
    options = list(options)
    durable = st.session_state.get(key)
    index = options.index(durable) if durable in options else int(kwargs.pop("index", 0))
    value = st.radio(label, options, index=min(index, len(options) - 1),
                     key=f"__w_{key}", **kwargs)
    st.session_state[key] = value
    return value


def p_check(label, key, **kwargs):
    value = st.checkbox(label, value=bool(st.session_state.get(key, False)),
                        key=f"__w_{key}", **kwargs)
    st.session_state[key] = value
    return value


def p_number(label, key, **kwargs):
    durable = st.session_state.get(key)
    if durable is None:
        durable = kwargs.pop("value", kwargs.get("min_value", 0))
    low, high = kwargs.get("min_value"), kwargs.get("max_value")
    if low is not None and durable < low:
        durable = low
    if high is not None and durable > high:
        durable = high
    value = st.number_input(label, value=durable, key=f"__w_{key}", **kwargs)
    st.session_state[key] = value
    return value


def p_text(label, key, **kwargs):
    value = st.text_input(label, value=str(st.session_state.get(key) or ""),
                          key=f"__w_{key}", **kwargs)
    st.session_state[key] = value
    return value


def p_textarea(label, key, **kwargs):
    value = st.text_area(label, value=str(st.session_state.get(key) or ""),
                         key=f"__w_{key}", **kwargs)
    st.session_state[key] = value
    return value


def go(step: int) -> None:
    st.session_state["step"] = step
    st.session_state["page"] = "study"
    st.rerun()


def go_examples() -> None:
    st.session_state["page"] = "examples"
    st.rerun()


def open_run(run_id: str) -> None:
    """Jump to the furthest stage an existing run has reached."""
    st.session_state["run_id"] = run_id
    st.session_state["page"] = "study"
    if st.session_state.get("flow") == "simple":
        st.session_state["step"] = 3
        st.rerun()
    flags = workspace.stage_flags(run_id)
    if flags["evaluated"]:
        step = 7
    elif flags["approved"] or flags["piloted"] or flags["classified"]:
        step = 6
    elif flags["sealed"]:
        step = 6
    elif flags["audited"]:
        step = 5
    else:
        step = 5
    st.session_state["step"] = step
    st.rerun()


def run_flags() -> dict:
    run_id = st.session_state.get("run_id")
    return workspace.stage_flags(run_id) if run_id else {}


def columns_chosen() -> bool:
    return bool(st.session_state.get("text_column")) and bool(
        st.session_state.get("benchmark_id_column")) and bool(
        st.session_state.get("label_column"))


def classes_defined() -> bool:
    classes = st.session_state.get("classes", [])
    valid = [c for c in classes if c.get("name", "").strip() and c.get("definition", "").strip()]
    return len(valid) >= 2


def simple_step_allowed(step: int) -> bool:
    if step <= 1:
        return True
    if step == 2:
        return bool(st.session_state.get("dataset_meta")
                    and st.session_state.get("benchmark_meta"))
    if step == 3:
        if st.session_state.get("run_id"):
            return True
        proposal = st.session_state.get("proposal") or {}
        problems = proposal.get("blocking") if isinstance(proposal, dict) else None
        return proposal is not None and not problems
    return False


def step_allowed(step: int) -> bool:
    if st.session_state.get("flow", "simple") == "simple":
        return simple_step_allowed(step)
    flags = run_flags()
    if step <= 1:
        return True
    if step == 2:
        return bool(st.session_state.get("dataset_meta")
                    and st.session_state.get("benchmark_meta")
                    and not flags.get("sealed"))
    if step == 3:
        return columns_chosen() and not flags.get("sealed")
    if step == 4:
        return columns_chosen() and classes_defined()
    if step == 5:
        return columns_chosen() and classes_defined()
    if step == 6:
        return bool(flags.get("sealed"))
    if step == 7:
        return bool(flags.get("frozen"))
    return False

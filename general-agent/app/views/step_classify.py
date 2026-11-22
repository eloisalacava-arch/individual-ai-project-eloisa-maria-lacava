"""Step 6 - blind pilot, approval, then the full classification."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import engine_adapter, workspace
from app.state import go, p_textarea


def _pilot_table(run_id: str) -> pd.DataFrame | None:
    path = workspace.run_dir(run_id) / "pilot" / "pilot_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    blind_path = workspace.run_dir(run_id) / "blind" / "blind.csv"
    if blind_path.exists():
        blind = pd.read_csv(blind_path, dtype="string", keep_default_na=False)
        frame = frame.merge(blind, on="id", how="left")
        frame["text"] = frame["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 90)
    return frame[["id", "text", "label", "status", "attempts"]]


def render() -> None:
    st.subheader("6 · Classify")
    st.caption("The pilot proves the prompt and parser work before the full run. "
               "The AI never sees a human label.")

    run_id = st.session_state.get("run_id")
    flags = workspace.stage_flags(run_id) if run_id else {}
    if not run_id or not flags.get("sealed"):
        st.warning("Seal the benchmark first.")
        return
    directory = workspace.run_dir(run_id)
    config_path = directory / "study.yaml"

    _budget_strip(run_id)
    _pilot_section(run_id, config_path, flags)
    _full_section(run_id, config_path, flags)


def _budget_strip(run_id: str) -> None:
    from general_agent.config import load_config
    config_path = workspace.run_dir(run_id) / "study.yaml"
    cfg = load_config(config_path)
    ceiling = int(cfg["budget"]["max_calls"])
    used = workspace.calls_used(run_id)
    cost = workspace.cost_totals(run_id)
    columns = st.columns(3)
    columns[0].metric("Calls used", f"{used}/{ceiling}")
    if cost["reported_calls"]:
        columns[1].metric("Provider-reported cost", f"${cost['actual_usd']:.4f}")
    else:
        columns[1].metric("Provider-reported cost", "not reported")
    columns[2].metric("Estimated floor", f"${cost['estimated_usd']:.4f}")


def _pilot_section(run_id: str, config_path, flags: dict) -> None:
    st.markdown("#### Pilot")
    if not flags["piloted"]:
        st.caption("Classifies the first five rows of the blind order. Five calls, "
                   "counted against the run budget.")
        if st.button("Run pilot (5 calls)", type="primary"):
            result = engine_adapter.pilot(config_path)
            st.session_state["last_stage"] = {"pilot": {
                "code": result.code, "log": result.log, "error": result.error}}
            st.rerun()
        _show_last("pilot")
        return

    table = _pilot_table(run_id)
    if table is not None:
        st.dataframe(table, width="stretch", hide_index=True)
        ok = int((table["status"] == "ok").sum())
        st.caption(f"{ok} of {len(table)} pilot rows parsed a declared category.")
    _show_last("pilot")

    approved, detail = engine_adapter.approval_status(config_path)
    if approved:
        st.success(f"Pilot approved — {detail}")
        if st.button("Re-run pilot"):
            result = engine_adapter.pilot(config_path)
            st.session_state["last_stage"] = {"pilot": {
                "code": result.code, "log": result.log, "error": result.error}}
            st.rerun()
        return

    st.warning(f"Full run blocked: {detail}")
    p_textarea("Approval note (optional)", "pilot_note", height=70)
    if st.button("Approve pilot", type="primary"):
        result = engine_adapter.approve(config_path, note=st.session_state["pilot_note"])
        st.session_state["last_stage"] = {"approve": {
            "code": result.code, "log": result.log, "error": result.error}}
        st.rerun()


def _full_section(run_id: str, config_path, flags: dict) -> None:
    st.markdown("#### Full run")
    approved, _ = engine_adapter.approval_status(config_path)
    if not approved:
        st.caption("Approve the pilot to unlock the full run.")
        return

    counts = workspace.predictions_counts(run_id)
    total, done = counts["total"], counts["ok"]
    st.progress(done / total if total else 0.0,
                text=f"{done} of {total} rows classified")
    if counts["statuses"]:
        chips = " ".join(f"{status}: {n}" for status, n in sorted(counts["statuses"].items()))
        st.caption(chips)

    if counts["total"] == 0 or counts["pending"] > 0:
        batch = int(st.session_state.get("batch_size", 25))
        columns = st.columns(3)
        if columns[0].button(f"Run next {batch} rows", type="primary"):
            _run(config_path, batch)
        if columns[1].button("Run all remaining"):
            _run(config_path, None)
        if columns[2].button("Resume"):
            _run(config_path, None)
        st.caption("Each batch is one engine invocation. Predictions are saved after "
                   "every row, so a closed tab loses at most the in-flight call.")
    else:
        if counts["failed"]:
            st.warning(f"{counts['failed']} row(s) did not return a usable category. "
                       "They are excluded from the metrics, never imputed. Resuming "
                       "retries them.")
            if st.button("Retry failed rows"):
                _run(config_path, None)
        else:
            st.success("All rows classified.")
        if st.button("Continue to results", type="primary"):
            go(7)
    _show_last("classify")


def _run(config_path, limit) -> None:
    result = engine_adapter.classify_batch(config_path, limit)
    st.session_state["last_stage"] = {"classify": {
        "code": result.code, "log": result.log, "error": result.error}}
    st.rerun()


def _show_last(name: str) -> None:
    entry = (st.session_state.get("last_stage") or {}).get(name)
    if not entry:
        return
    if entry.get("error"):
        st.error(entry["error"])
    with st.expander(f"Last {name} log", expanded=entry.get("code") not in (0,)):
        st.code(entry.get("log", ""), language=None)

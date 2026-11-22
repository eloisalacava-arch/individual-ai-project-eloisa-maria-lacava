"""Step 5 - create the study, validate it, and seal the benchmark."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app import config_builder, engine_adapter, registry, workspace
from app.state import go, p_text, p_textarea

CHECK_HELP = {
    "dataset_excludes_reference_label_columns":
        "The observations file appears to contain the human label. Keep the two "
        "files separate; the classifier must never see a label column.",
    "id_sets_match": "IDs do not match between the two files.",
    "id_sets_match_after_policy": "IDs still do not match after applying the policy.",
    "all_declared_classes_present_in_primary":
        "A declared category never appears in the mapped human labels.",
    "dataset_text_nonblank": "Some observation texts are empty.",
    "dataset_ids_unique": "The dataset has duplicate IDs.",
    "benchmark_ids_unique": "The benchmark has duplicate IDs.",
    "dataset_ids_complete": "The dataset has blank or missing IDs.",
    "benchmark_ids_complete": "The benchmark has blank or missing IDs.",
}


def _spec(run_id: str, materialised: dict) -> dict:
    from app.session_spec import build_spec
    return build_spec(run_id, materialised)


def render() -> None:
    st.subheader("5 · Validate and seal")
    st.caption("Validation runs the engine audit. Sealing splits the data so the "
               "classifier can only read the id and the text.")

    with st.expander("Prompt and report details", expanded=False):
        p_text("Task role (one sentence the AI sees)", "task_role")
        p_textarea("Decision rule (optional)", "decision_rule", height=80)
        p_text("Report title", "report_title")
        p_text("Report author", "report_author")
        p_text("Task description (for the report)", "task_description")
        p_text("Reference description (for the report)", "reference_description")

    run_id = st.session_state.get("run_id")
    flags = workspace.stage_flags(run_id) if run_id else {}
    locked = bool(flags.get("sealed") or flags.get("piloted") or flags.get("classified"))

    if st.button("Create study and validate", type="primary"):
        _create_and_validate()

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.caption("Nothing is written until you validate.")
        return

    directory = workspace.run_dir(run_id)
    st.markdown(f"**Run directory** `{workspace.display_path(directory)}`")
    if locked:
        st.info("This run is sealed; re-validation is disabled to protect it.")
    _show_validation(run_id)
    _seal_section(run_id)


def _create_and_validate() -> None:
    run_id = st.session_state.get("run_id")
    flags = workspace.stage_flags(run_id) if run_id else {}
    if flags.get("sealed") or flags.get("classified"):
        st.error("This run is already sealed or classified. Start a new study from "
                 "the sidebar to change the data or categories.")
        return
    if not run_id:
        run_id = workspace.next_run_id(st.session_state.get("study_name") or "study")
        st.session_state["run_id"] = run_id
    directory = workspace.run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)

    materialised = workspace.materialise_inputs(
        directory,
        st.session_state["dataset_upload"],
        st.session_state["benchmark_upload"],
        st.session_state["dataset_meta"],
        make_id=st.session_state["make_id"],
    )
    if materialised["row_note"]:
        st.caption(materialised["row_note"])

    spec = _spec(run_id, materialised)
    problems = config_builder.validate_spec(spec)
    if problems:
        for problem in problems:
            st.error(problem)
        return

    try:
        config = config_builder.build_config(spec)
        config_path = config_builder.write_config(directory, config)
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        return

    configure_result = engine_adapter.configure(config_path)
    audit_result = engine_adapter.audit(config_path)
    st.session_state["last_stage"] = {
        "configure": {"code": configure_result.code, "log": configure_result.log,
                      "error": configure_result.error},
        "audit": {"code": audit_result.code, "log": audit_result.log,
                  "error": audit_result.error},
    }


def _show_validation(run_id: str) -> None:
    decisions = workspace.read_decisions(run_id)
    manifest = workspace.load_manifest(run_id)
    last = st.session_state.get("last_stage") or {}
    audit = last.get("audit")

    if audit and audit.get("error"):
        st.error(audit["error"])
    if audit:
        with st.expander("Audit log", expanded=audit.get("code") != 0):
            st.code(audit.get("log", ""), language=None)

    checks = manifest.get("checks") or {}
    if checks:
        failed = checks.get("failed") or []
        column_left, column_right, column_third = st.columns(3)
        column_left.metric("Checks passed",
                           f"{checks.get('passed', 0)}/{checks.get('total', 0)}")
        column_right.metric("Warnings", len(manifest.get("warnings") or []))
        column_third.metric("Status", checks.get("status", "?"))
        if failed:
            st.error("Failed checks")
            for name in failed:
                st.markdown(f"- `{name}` — {CHECK_HELP.get(name, 'See the audit log.')}")

    warnings = manifest.get("warnings") or []
    if warnings:
        with st.expander(f"Warnings ({len(warnings)})"):
            for warning in warnings:
                st.markdown(f"- {warning}")

    questions = decisions.get("questions") or []
    if decisions.get("status") == "needs_clarification" and questions:
        st.warning("The engine needs a decision before sealing.")
        for question in questions:
            st.markdown(f"**{question['field']}** — {question['reason']}")
            st.caption(f"Resolve by: {question['how_to_resolve']}")
        columns = st.columns(2)
        if columns[0].button("Back to columns"):
            go(2)
        if columns[1].button("Back to categories"):
            go(3)

    accounting = manifest.get("row_accounting") or []
    if accounting:
        with st.expander(f"Row accounting ({len(accounting)} step(s))"):
            st.dataframe(pd.DataFrame(accounting), width="stretch",
                         hide_index=True)


def _seal_section(run_id: str) -> None:
    flags = workspace.stage_flags(run_id)
    directory = workspace.run_dir(run_id)
    decisions = workspace.read_decisions(run_id)
    manifest = workspace.load_manifest(run_id)
    audit_pass = (manifest.get("checks") or {}).get("status") == "PASS"

    if not flags["sealed"]:
        if not audit_pass or decisions.get("status") != "resolved":
            st.caption("Fix the failed checks and any questions, then validate again.")
            return
        if st.button("Seal benchmark and continue"):
            config_path = directory / "study.yaml"
            result = engine_adapter.seal(config_path)
            st.session_state["last_stage"] = {"seal": {
                "code": result.code, "log": result.log, "error": result.error}}
            st.rerun()
        return

    st.success("Benchmark sealed. Human labels are hidden until evaluation.")
    info_path = directory / "blind_public_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        sample = info.get("sample", {})
        st.markdown(
            f"<span class='ga-muted'>rows sealed: {info.get('n_selected')} · "
            f"method {sample.get('method')} · n {sample.get('requested_n')} · "
            f"seed {info.get('seed')} · stratified {sample.get('stratified')}</span>",
            unsafe_allow_html=True)
    _prompt_preview(directory)

    piloted = flags["piloted"]
    label = "Continue to classification" if not piloted else "Back to classification"
    if st.button(label, type="primary"):
        go(6)


def _prompt_preview(directory) -> None:
    from general_agent.config import load_config
    from general_agent.prompts import build_prompt, prompt_template_sha
    import pandas as pd
    blind_path = directory / "blind" / "blind.csv"
    if not blind_path.exists():
        return
    payload = pd.read_csv(blind_path, dtype="string", keep_default_na=False)
    ids = payload["id"].tolist()
    st.markdown("#### Exactly what the AI will receive")
    st.caption("This is built by the engine's own prompt function. The human label "
               "is not present.")
    selected = st.selectbox("Preview a row", ids, key="preview_id") if ids else None
    if selected is None:
        return
    text = payload.loc[payload["id"] == selected, "text"].iloc[0]
    cfg = load_config(directory / "study.yaml")
    prompt = build_prompt(cfg, text)
    st.code(prompt, language=None)
    st.caption(f"Prompt template SHA-256: {prompt_template_sha(cfg)[:24]}…")

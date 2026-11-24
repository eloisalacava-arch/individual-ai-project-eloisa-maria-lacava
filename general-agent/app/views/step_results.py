"""Step 7 - evaluate, read the results, and download everything."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import (
    engine_adapter, qualitative_support, registry, report_support, result_view,
    workspace,
)
from app.state import go, p_check, p_number, p_select


def render() -> None:
    st.subheader("7 · Results")
    run_id = st.session_state.get("run_id")
    flags = workspace.stage_flags(run_id) if run_id else {}
    if not run_id or not flags.get("sealed"):
        st.warning("Classify first.")
        return
    directory = workspace.run_dir(run_id)
    config_path = directory / "study.yaml"

    if not flags["evaluated"]:
        counts = workspace.predictions_counts(run_id)
        if counts["pending"] > 0:
            st.warning(f"{counts['pending']} row(s) are still pending. Classify them "
                       "first, or evaluate the ones already done.")
        if st.button("Evaluate predictions and generate report", type="primary"):
            _evaluate(run_id, config_path)
            st.rerun()
        _show_last()
        return

    frames = result_view.load_results(directory)
    _headline(frames)
    _confusion(directory)
    _per_class(frames)
    _distribution(frames)
    _disagreements(directory)
    _downloads(run_id, directory)
    _method_notes(run_id, config_path)
    _qualitative_section(run_id, directory, config_path)
    _reports_section(run_id)
    _show_last()


def _evaluate(run_id: str, config_path) -> None:
    logs = {}
    logs["evaluate"] = engine_adapter.evaluate(config_path)
    if (workspace.run_dir(run_id) / "results" / "t02_accuracy.csv").exists():
        logs["disagreements"] = engine_adapter.disagreements(config_path)
        logs["report"] = engine_adapter.report(
            config_path, workspace.run_dir(run_id) / "report")
    st.session_state["last_stage"] = {"evaluation": {
        name: {"code": result.code, "log": result.log, "error": result.error}
        for name, result in logs.items()}}


def _headline(frames: dict) -> None:
    accuracy = result_view.accuracy_row(frames)
    if not accuracy:
        return
    columns = st.columns(4)
    columns[0].metric("Rows analysed", f"{int(accuracy.get('n', 0))}")
    columns[1].metric(
        "Accuracy",
        result_view.fmt(accuracy.get("accuracy"), 3),
        help="Exact matches. 95% Wilson interval "
             f"{result_view.fmt(accuracy.get('accuracy_ci_low'))}–"
             f"{result_view.fmt(accuracy.get('accuracy_ci_high'))}.")
    columns[2].metric(
        "Balanced accuracy",
        result_view.fmt(accuracy.get("balanced_accuracy"), 3),
        help="Average per-class recall; bootstrap interval "
             f"{result_view.fmt(accuracy.get('ba_ci_low'))}–"
             f"{result_view.fmt(accuracy.get('ba_ci_high'))}.")
    kappas = result_view.kappa_rows(frames)
    unweighted = next((row for row in kappas if "unweighted" in row["metric"]), None)
    columns[3].metric(
        "Cohen's kappa",
        result_view.fmt(unweighted["value"]) if unweighted else "not estimable",
        help="Chance-corrected agreement. Depends on the class distribution.")
    st.markdown(
        "<span class='ga-metric-source'>Source: results/t02_accuracy.csv, "
        "results/t03_kappa.csv — scripts/05_evaluate.py</span>",
        unsafe_allow_html=True)
    for row in kappas:
        if "quadratic" in row["metric"]:
            st.caption(f"Quadratic weighted kappa: {result_view.fmt(row['value'])} "
                       f"({result_view.fmt(row['ci_low'])}–"
                       f"{result_view.fmt(row['ci_high'])})")


def _confusion(directory) -> None:
    matrix, _ = result_view.confusion_frames(directory)
    if matrix is None or matrix.empty:
        return
    st.markdown("#### Confusion matrix")
    st.caption("Rows are the human label; columns are the AI answer. The diagonal "
               "is agreement.")
    display = matrix.rename_axis("human \\\\ AI", axis=0).rename_axis("AI answer", axis=1)
    try:
        styled = display.style.background_gradient(cmap="Blues").format("{:d}")
        st.dataframe(styled, width="stretch")
    except Exception:
        st.dataframe(display, width="stretch")


def _per_class(frames: dict) -> None:
    frame = result_view.per_class_rows(frames)
    if frame is None or frame.empty:
        return
    st.markdown("#### By category")
    st.caption("Precision, recall and F1 with 95% intervals. Support is the number "
               "of human-labelled rows in the category.")
    columns = [c for c in ["class", "n_reference", "n_predicted", "precision",
                           "recall", "f1"] if c in frame.columns]
    st.dataframe(frame[columns], width="stretch", hide_index=True)


def _distribution(frames: dict) -> None:
    frame = result_view.distribution_rows(frames)
    if frame is None or frame.empty:
        return
    with st.expander("Class distribution (human vs AI)"):
        st.dataframe(frame, width="stretch", hide_index=True)


def _disagreements(directory) -> None:
    frame = result_view.text_disagreements(directory)
    if frame is None or frame.empty:
        st.success("No disagreements between the AI and the human labels.")
        return
    st.markdown(f"#### Disagreements ({len(frame)})")
    st.caption("A disagreement is not a confirmed model error: the human labels are "
               "a benchmark, not adjudicated truth.")
    references = ["all"] + sorted(frame["reference"].unique())
    predictions = ["all"] + sorted(frame["predicted"].unique())
    left, middle, right = st.columns(3)
    reference = left.selectbox("Human label", references, key="filter_reference")
    prediction = middle.selectbox("AI answer", predictions, key="filter_predicted")
    severities = ["all"] + sorted(frame["severity"].unique()) if "severity" in frame else ["all"]
    severity = right.selectbox("Severity", severities, key="filter_severity")
    view = frame
    if reference != "all":
        view = view[view["reference"] == reference]
    if prediction != "all":
        view = view[view["predicted"] == prediction]
    if severity != "all" and "severity" in view:
        view = view[view["severity"] == severity]
    st.caption(f"{len(view)} of {len(frame)} disagreement(s) shown")
    st.dataframe(view, width="stretch", hide_index=True,
                 column_config={"text": st.column_config.TextColumn(width="large")})


def _downloads(run_id: str, directory) -> None:
    st.markdown("#### Downloads")
    report = directory / "report" / "report.md"
    predictions = result_view.prediction_rows(directory)
    prediction_path = None
    for candidate in (directory / "results" / "frozen_predictions.csv",
                      directory / "predictions.csv"):
        if candidate.exists():
            prediction_path = candidate
            break

    results_zip = result_view.make_zip(
        directory, result_view.result_files(directory))
    provenance_zip = result_view.make_zip(
        directory, result_view.provenance_files(directory))

    columns = st.columns(2)
    if report.exists():
        columns[0].download_button(
            "Generated report (Markdown)", data=report.read_bytes(),
            file_name=f"report_{run_id}.md", mime="text/markdown")
    if prediction_path:
        columns[1].download_button(
            "Predictions (CSV)",
            data=prediction_path.read_bytes(),
            file_name=f"predictions_{run_id}.csv", mime="text/csv")
    columns[0].download_button(
        "Result tables (ZIP)", data=results_zip,
        file_name=f"results_{run_id}.zip", mime="application/zip")
    columns[1].download_button(
        "Provenance bundle (ZIP)", data=provenance_zip,
        file_name=f"provenance_{run_id}.zip", mime="application/zip")
    st.caption("The provenance bundle contains the config, manifest, decisions, "
               "attempt logs (full prompts and outputs), and results. The sealed "
               "benchmark is not included.")


def _method_notes(run_id: str, config_path) -> None:
    from pathlib import Path

    from general_agent.config import load_config
    with st.expander("Method notes and limitations"):
        st.markdown("- Accuracy and recall use Wilson intervals; kappa, balanced "
                    "accuracy, precision and F1 use seeded percentile bootstrap.")
        st.markdown("- No causal claims: these are associations on one sample.")
        if Path(config_path).exists():
            cfg = load_config(config_path)
            description = cfg.get("report", {}).get("reference_description")
            if description:
                st.markdown(f"- Reference: {description}")
            for limitation in cfg.get("report", {}).get("limitations", []):
                st.markdown(f"- {limitation}")
            st.markdown(f"- Random seed: {cfg['run']['seed']}. Model: "
                        f"`{cfg['model']['name']}`.")
        else:
            st.markdown("- This run was produced by the command line; its study "
                        "configuration is not stored in the run directory.")
        st.markdown("- Every number on this page is read from "
                    "`runs/" + run_id + "/results/`.")


def _qualitative_section(run_id: str, directory, config_path) -> None:
    st.markdown("#### Qualitative analysis (AI hypotheses)")
    st.caption("Explains possible reasons the AI and the human might differ, using "
               "a model. These are hypotheses, not confirmed causes and not proof "
               "of a model error. The statistical results above are unaffected.")
    from pathlib import Path
    if not Path(config_path).exists():
        st.caption("This run was produced by the command line and its study "
                   "configuration is not in the run directory, so the qualitative "
                   "section is unavailable here. Use a study created in this "
                   "interface, or the CLI stage 08.")
        return
    disagreements_path = directory / "results" / "t08_disagreements.csv"
    if not disagreements_path.exists():
        st.info("Run the evaluation first; the disagreement set comes from stage 05.")
        return
    n_disagreements = len(pd.read_csv(disagreements_path, dtype="string"))
    if n_disagreements == 0:
        st.success("No disagreements, so there is nothing to explain.")
        return

    explanations = result_view.qualitative_explanations(directory)
    pilot = result_view.qualitative_pilot_explanations(directory)
    sidecar = qualitative_support.settings_path(directory)

    _qualitative_form(run_id, directory, config_path, n_disagreements,
                      locked=explanations is not None)

    if pilot is not None:
        st.markdown("**Pilot explanations (review before approving)**")
        st.dataframe(pilot[_pilot_columns(pilot)], width="stretch", hide_index=True)

    approved, detail = (False, "run the pilot first")
    if sidecar.exists():
        approved, detail = engine_adapter.qualitative_approval_status(
            config_path, sidecar)

    if explanations is None:
        if pilot is None:
            if st.button("Explain a 3-item pilot", type="primary"):
                _run_qualitative_pilot(config_path, sidecar)
        elif not approved:
            st.warning(f"Full run blocked: {detail}")
            if st.button("Approve pilot and explain the rest", type="primary"):
                result = engine_adapter.qualitative_approve(
                    config_path, sidecar, note=st.session_state.get("qu_note", ""))
                st.session_state["last_stage"] = {"qualitative": {
                    "code": result.code, "log": result.log, "error": result.error}}
                st.rerun()
        else:
            _qualitative_run_controls(run_id, directory, config_path, sidecar)
        return

    st.success(f"Explanations present for {len(explanations)} selected item(s).")
    _qualitative_run_controls(run_id, directory, config_path, sidecar)
    _qualitative_findings(run_id, directory, config_path, sidecar)


def _pilot_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["group", "id", "reference", "predicted", "reasons",
                        "summary", "confidence", "status"] if c in frame.columns]


def _qualitative_form(run_id: str, directory, config_path, n_disagreements: int,
                      locked: bool) -> None:
    from general_agent.config import load_config
    providers = registry.enabled_providers()
    if not providers:
        st.error("No provider is enabled in configs/ui_providers.yaml.")
        return
    cfg = load_config(config_path)

    with st.expander("Qualitative settings", expanded=not locked):
        if locked:
            st.caption("Settings are fixed once explanations exist. Start a new "
                       "study to change them.")
        provider_ids = [provider["id"] for provider in providers]
        st.session_state.setdefault("qu_provider_id", provider_ids[0])
        provider_id = p_select(
            "Provider", provider_ids, "qu_provider_id", disabled=locked,
            format_func=lambda value: next(
                provider["label"] for provider in providers
                if provider["id"] == value))
        provider = registry.find_provider(provider_id)
        models = provider.get("models", [])
        model_names = [model["name"] for model in models]
        st.session_state.setdefault("qu_model_name", model_names[0])
        model_name = p_select(
            "Model", model_names, "qu_model_name", disabled=locked,
            format_func=lambda value: next(
                model.get("label", model["name"]) for model in models
                if model["name"] == value))
        model = registry.find_model(provider_id, model_name)
        key_env, present = registry.credential_status(model, provider)
        if key_env and not present:
            st.warning(registry.missing_key_message(key_env))

        st.session_state.setdefault("qu_sample_n", min(25, max(n_disagreements, 1)))
        st.session_state.setdefault("qu_control_n", 10)
        st.session_state.setdefault("qu_seed", int(cfg["run"]["seed"]))
        st.session_state.setdefault("qu_max_calls", 45)
        st.session_state.setdefault("qu_max_usd", 0.25)
        columns = st.columns(2)
        with columns[0]:
            p_number("Disagreements to explain", "qu_sample_n", min_value=1,
                     max_value=max(n_disagreements, 1), disabled=locked)
        with columns[1]:
            p_number("Control (agreements)", "qu_control_n", min_value=0,
                     max_value=50, disabled=locked)
        columns = st.columns(2)
        with columns[0]:
            p_number("Seed", "qu_seed", min_value=1, step=1, disabled=locked)
        with columns[1]:
            p_number("Maximum calls", "qu_max_calls", min_value=3, max_value=200,
                     disabled=locked)
        p_number("Maximum spend (USD)", "qu_max_usd", min_value=0.0, step=0.01,
                 format="%.2f", disabled=locked)
        p_check("I understand the disagreement text and the human label will "
                "be sent to this provider.", "qu_consent", disabled=locked)

        calls = int(st.session_state["qu_sample_n"]) + int(
            st.session_state["qu_control_n"]) + 3
        floor = qualitative_support.estimate_floor_usd(model, calls)
        st.caption(f"Minimum calls: {qualitative_support.minimum_calls(st.session_state['qu_sample_n'], st.session_state['qu_control_n'])} "
                   f"(pilot included). Approximate cost floor: ${floor:.4f}. "
                   "Provider-reported cost is what is recorded.")


def _current_sidecar(directory, config_path) -> dict | None:
    from general_agent.config import load_config
    providers = registry.enabled_providers()
    provider = registry.find_provider(st.session_state.get("qu_provider_id"))
    model = registry.find_model(st.session_state.get("qu_provider_id"),
                                st.session_state.get("qu_model_name"))
    if not provider or not model:
        return None
    return qualitative_support.build_sidecar(
        provider, model,
        sample_n=int(st.session_state["qu_sample_n"]),
        control_n=int(st.session_state["qu_control_n"]),
        seed=int(st.session_state["qu_seed"]),
        max_calls=int(st.session_state["qu_max_calls"]),
        max_usd=float(st.session_state["qu_max_usd"]),
    )


def _run_qualitative_pilot(config_path, sidecar_path) -> None:
    if not st.session_state.get("qu_consent"):
        st.error("Tick the data-sharing box before running the pilot.")
        return
    sidecar = _current_sidecar(None, config_path)
    if sidecar is None:
        st.error("Choose a provider and model.")
        return
    qualitative_support.write_sidecar(workspace.run_dir(st.session_state["run_id"]),
                                      sidecar)
    result = engine_adapter.qualitative_pilot(config_path, sidecar_path)
    st.session_state["last_stage"] = {"qualitative": {
        "code": result.code, "log": result.log, "error": result.error}}
    st.rerun()


def _qualitative_run_controls(run_id: str, directory, config_path, sidecar) -> None:
    if not sidecar.exists():
        st.caption("Run the pilot to create the settings file.")
        return
    explanations = result_view.qualitative_explanations(directory)
    if explanations is not None:
        pending = int((explanations["status"] != "ok").sum())
        total = len(explanations)
        st.progress((total - pending) / total if total else 0.0,
                    text=f"{total - pending} of {total} explained")
        stage = result_view.qualitative_stage(directory)
        calls = result_view.qualitative_calls_used(directory)
        cap = stage.get("max_calls", "?")
        cost = stage.get("actual_usd")
        columns = st.columns(2)
        columns[0].metric("Explanation calls", f"{calls}/{cap}")
        columns[1].metric("Provider-reported cost",
                          f"${cost:.4f}" if isinstance(cost, (int, float)) and calls
                          else "not reported")
    else:
        pending = None

    approved, _detail = engine_adapter.qualitative_approval_status(config_path, sidecar)
    if not approved:
        st.caption("Approve the pilot to unlock the full run.")
        return
    columns = st.columns(3)
    if columns[0].button("Explain next 10"):
        result = engine_adapter.qualitative_run(config_path, sidecar, limit=10)
        st.session_state["last_stage"] = {"qualitative": {
            "code": result.code, "log": result.log, "error": result.error}}
        st.rerun()
    if columns[1].button("Explain all remaining"):
        result = engine_adapter.qualitative_run(config_path, sidecar, limit=None)
        st.session_state["last_stage"] = {"qualitative": {
            "code": result.code, "log": result.log, "error": result.error}}
        st.rerun()
    if columns[2].button("Generate qualitative report"):
        result = engine_adapter.qualitative_report(config_path, sidecar)
        st.session_state["last_stage"] = {"qualitative": {
            "code": result.code, "log": result.log, "error": result.error}}
        st.rerun()


def _qualitative_findings(run_id: str, directory, config_path, sidecar) -> None:
    frames = result_view.load_qualitative(directory)
    if "q05_coverage.csv" in frames:
        st.markdown("**Coverage**")
        st.dataframe(frames["q05_coverage.csv"], width="stretch", hide_index=True)
    if "q02_patterns.csv" in frames:
        patterns = frames["q02_patterns.csv"]
        for group, label in (("disagreement", "Disagreements"),
                             ("agreement", "Control (agreements)")):
            subset = patterns[patterns["group"] == group]
            if subset.empty:
                continue
            st.markdown(f"**Hypothesised reasons — {label}**")
            st.dataframe(subset, width="stretch", hide_index=True)
    if "q07_reason_contrast.csv" in frames:
        st.markdown("**Contrast with the control sample**")
        st.caption("A reason common in both groups is not specific to disagreements.")
        st.dataframe(frames["q07_reason_contrast.csv"], width="stretch",
                     hide_index=True)
    if "q04_examples.csv" in frames:
        examples = frames["q04_examples.csv"]
        examples = examples[examples["group"] == "disagreement"]
        if not examples.empty:
            st.markdown("**Illustrative examples (highest-confidence per reason)**")
            st.dataframe(examples, width="stretch", hide_index=True,
                         column_config={"text_excerpt": st.column_config.TextColumn(
                             width="large")})

    narrative = result_view.qualitative_narrative(directory)
    if narrative:
        st.markdown("**Narrative report**")
        with st.expander("Preview the narrative report"):
            st.markdown(narrative)
    _qualitative_downloads(run_id, directory, sidecar)


def _qualitative_downloads(run_id: str, directory, sidecar) -> None:
    narrative_path = result_view.qualitative_dir(directory) / "narrative_report.md"
    explanations_path = result_view.qualitative_dir(directory) / "explanations.csv"
    columns = st.columns(3)
    if narrative_path.exists():
        columns[0].download_button(
            "Narrative report (Markdown)", data=narrative_path.read_bytes(),
            file_name=f"qualitative_{run_id}.md", mime="text/markdown")
    if explanations_path.exists():
        columns[1].download_button(
            "Explanations (CSV)", data=explanations_path.read_bytes(),
            file_name=f"explanations_{run_id}.csv", mime="text/csv")
    files = result_view.qualitative_files(directory)
    if files:
        columns[2].download_button(
            "Qualitative bundle (ZIP)",
            data=result_view.make_zip(result_view.qualitative_dir(directory), files),
            file_name=f"qualitative_{run_id}.zip", mime="application/zip")
    st.caption("The bundle includes the settings, approval, pilot, attempt log "
               "(full prompts and outputs), tables, and narrative.")


def _reports_section(run_id: str) -> None:
    st.markdown("#### Full report")
    st.caption("One document for this study: the statistical results and, when "
               "available, the qualitative hypotheses. Composed from the result "
               "files only; no model call.")
    if not report_support.config_available(run_id):
        st.info("No study configuration was found for this run, so a full report "
                "cannot be composed here.")
        return

    study = report_support.study_report_paths(run_id)
    qualitative_pdf = report_support.qualitative_pdf_path(run_id)
    included = report_support.qualitative_included(run_id)

    if not included:
        st.info("Qualitative analysis has not been run for this study. The report "
                "states this explicitly and includes no explanations.")

    columns = st.columns(2)
    label = "Regenerate full report" if study["markdown"].exists() else \
        "Generate full report"
    if columns[0].button(label, type="primary"):
        with st.spinner("Composing the report…"):
            try:
                report_support.build_study(run_id)
            except Exception as exc:
                st.error(f"Could not build the report: {type(exc).__name__}: {exc}")
            else:
                st.rerun()

    if not report_support.pdf_available():
        st.caption("PDF export needs ReportLab: `uv sync --extra pdf`. A Markdown "
                   "report is still available.")

    columns = st.columns(3)
    if study["markdown"].exists():
        columns[0].download_button(
            "Full report (Markdown)", data=study["markdown"].read_bytes(),
            file_name=f"study_report_{run_id}.md", mime="text/markdown")
    if study["pdf"].exists():
        columns[1].download_button(
            "Download Full Report (PDF)", data=study["pdf"].read_bytes(),
            file_name=f"study_report_{run_id}.pdf", mime="application/pdf")
    if qualitative_pdf.exists():
        columns[2].download_button(
            "Qualitative report (PDF)", data=qualitative_pdf.read_bytes(),
            file_name=f"qualitative_report_{run_id}.pdf", mime="application/pdf")
    if not study["markdown"].exists():
        st.caption("Generate the report to enable the downloads.")


def _show_last() -> None:
    last = st.session_state.get("last_stage") or {}
    entry = last.get("evaluation")
    if entry:
        with st.expander("Last evaluation log", expanded=False):
            for name, result in entry.items():
                st.markdown(f"**{name}** — exit code {result['code']}")
                if result.get("error"):
                    st.error(result["error"])
                st.code(result.get("log", ""), language=None)
    qualitative = last.get("qualitative")
    if qualitative:
        with st.expander("Last qualitative log", expanded=qualitative.get("code") not in (0,)):
            if qualitative.get("error"):
                st.error(qualitative["error"])
            st.code(qualitative.get("log", ""), language=None)

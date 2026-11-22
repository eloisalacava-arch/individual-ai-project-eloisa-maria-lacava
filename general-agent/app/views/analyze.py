"""Step 3 (simple) - one authorization, then the whole bounded analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app import (
    config_builder, pipeline, qualitative_support, registry, report_support,
    result_view, session_spec, workspace,
)
from general_agent.provenance import now_iso
from app.state import go, p_check


def render() -> None:
    st.subheader("3 · Analyze & Download")
    run_id = st.session_state.get("run_id")
    authorization = st.session_state.get("authorization")
    if run_id and not authorization:
        authorization = workspace.read_authorization(run_id)
        st.session_state["authorization"] = authorization

    if not authorization:
        _authorize_form()
        return
    _drive(run_id, authorization)


# --------------------------------------------------------------------------- #
# Authorization                                                               #
# --------------------------------------------------------------------------- #

def _authorize_form() -> None:
    st.caption("Nothing is sent anywhere until you authorize the run below.")
    if st.session_state.get("proposal", {}).get("blocking"):
        st.error("Some decisions are still missing. Go back to Review.")
        if st.button("Back to review"):
            go(2)
        return

    rows = int((st.session_state.get("dataset_meta") or {}).get("n_rows", 0))
    if st.session_state.get("rows_mode") == "subset":
        rows = min(rows, int(st.session_state.get("sample_n") or rows))
    calls = int(st.session_state.get("max_calls") or 0)
    usd = float(st.session_state.get("max_usd") or 0)
    qualitative = bool(st.session_state.get("qualitative_enabled"))
    q_calls = int(st.session_state.get("qualitative_max_calls") or 0)
    q_usd = float(st.session_state.get("qualitative_max_usd") or 0)

    with st.container(border=True):
        st.markdown("#### What will run")
        st.markdown(
            f"- Validate the data and seal the human labels.\n"
            f"- Five-row pilot, then classify up to {rows} observation(s) blind.\n"
            f"- Evaluate against the benchmark, and build the PDF report.\n"
            + ("- Explain disagreements with the AI when any are found "
               "(hypotheses only).\n" if qualitative else ""))

    combined_calls = calls + (q_calls if qualitative else 0)
    combined_usd = usd + (q_usd if qualitative else 0)
    with st.container(border=True):
        st.markdown("#### Budget (maximum before you authorize)")
        budget_rows = [
            "| Part | Model calls | Spend |",
            "|---|---:|---:|",
            f"| Classification (five-row pilot + up to {rows} rows) | {calls} "
            f"| ${usd:.2f} |",
        ]
        if qualitative:
            budget_rows.append(
                f"| Optional qualitative explanations | {q_calls} | ${q_usd:.2f} |")
        else:
            budget_rows.append(
                "| Optional qualitative explanations | not enabled | — |")
        budget_rows.append(
            f"| **Combined maximum** | **{combined_calls}** | **${combined_usd:.2f}** |")
        st.markdown("\n".join(budget_rows))
        st.caption(
            "The classification cap is enforced by the engine. Qualitative "
            "explanations are optional and run only when disagreements exist; their "
            "cap is separate. The combined maximum is the most this run can spend.")

    p_check("I authorize this bounded run: the model calls, the spend cap, and the "
            "stages described above.", "authorize_ack")
    if st.button("Authorize and run", type="primary",
                 disabled=not st.session_state.get("authorize_ack")):
        if _create_run() is not None:
            st.rerun()


def _create_run() -> str | None:
    run_id = st.session_state.get("run_id") or workspace.next_run_id(
        st.session_state.get("study_name") or "study")
    st.session_state["run_id"] = run_id
    directory = workspace.run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    materialised = workspace.materialise_inputs(
        directory, st.session_state["dataset_upload"],
        st.session_state["benchmark_upload"], st.session_state["dataset_meta"],
        make_id=st.session_state.get("make_id", False))
    if materialised["row_note"]:
        st.caption(materialised["row_note"])
    try:
        spec = session_spec.build_spec(run_id, materialised)
    except ValueError as exc:
        st.error(str(exc))
        return None
    problems = config_builder.validate_spec(spec)
    if problems:
        for problem in problems:
            st.error(problem)
        return None
    try:
        config = config_builder.build_config(spec)
        config_path = config_builder.write_config(directory, config)
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        return None

    from general_agent.config import load_config
    cfg = load_config(config_path)
    provider = registry.find_provider(st.session_state["provider_id"])
    model = registry.find_model(st.session_state["provider_id"],
                                st.session_state["model_name"])
    qualitative_enabled = bool(st.session_state.get("qualitative_enabled"))
    qualitative_settings = qualitative_support.build_sidecar(
        provider, model,
        sample_n=int(st.session_state.get("qualitative_sample_n") or 15),
        control_n=int(st.session_state.get("qualitative_control_n") or 5),
        seed=int(st.session_state.get("seed") or 20261116),
        max_calls=int(st.session_state.get("qualitative_max_calls") or 30),
        max_usd=float(st.session_state.get("qualitative_max_usd") or 0.25))
    calls = int(st.session_state.get("max_calls") or 0)
    usd = float(st.session_state.get("max_usd") or 0)
    q_calls = int(st.session_state.get("qualitative_max_calls") or 0)
    q_usd = float(st.session_state.get("qualitative_max_usd") or 0)
    authorization = {
        "run_id": run_id,
        "authorized_at": now_iso(),
        "provider": provider["id"],
        "model": model["name"],
        "config_sha256": cfg["_meta"]["config_sha256"],
        "classification": {"max_calls": calls, "max_usd": usd},
        "qualitative": {"enabled": qualitative_enabled,
                        "settings": qualitative_settings,
                        "max_calls": q_calls, "max_usd": q_usd},
        "total_max_calls": calls + (q_calls if qualitative_enabled else 0),
        "total_max_usd": round(usd + (q_usd if qualitative_enabled else 0), 2),
        "batch_size": int(st.session_state.get("batch_size") or 25),
        "allow_unclean_pilot": False,
    }
    workspace.write_authorization(run_id, authorization)
    st.session_state["authorization"] = authorization
    st.session_state["stop_requested"] = False
    st.session_state["auto_run"] = True
    st.session_state["pipeline_steps"] = 0
    return run_id


# --------------------------------------------------------------------------- #
# Progress                                                                    #
# --------------------------------------------------------------------------- #

STAGE_ORDER = ["validate", "seal", "pilot", "classify", "evaluate",
               "disagreements", "qualitative", "report"]


def _drive(run_id: str, authorization: dict) -> None:
    progress = workspace.progress(run_id)
    flags = progress["flags"]
    counts = progress["counts"]
    directory = workspace.run_dir(run_id)
    config_path = directory / "study.yaml"

    stopped = bool(st.session_state.get("stop_requested"))
    if not stopped:
        advance = pipeline.advance(run_id, config_path, authorization,
                                   stop=False,
                                   batch_size=int(authorization.get("batch_size") or 25))
        st.session_state["last_advance"] = {
            "status": advance.status, "stage": advance.stage,
            "action": advance.action, "detail": advance.detail,
            "log": advance.log[-4000:], "extra": advance.extra}
        progress = workspace.progress(run_id)
        flags = progress["flags"]
        counts = progress["counts"]
    else:
        advance = None

    last = st.session_state.get("last_advance") or {}
    if last.get("action"):
        st.caption(f"Latest action: {last['action']}")

    columns = st.columns(4)
    columns[0].metric("Rows classified", f"{counts['ok']} / {counts['total']}")
    columns[1].metric("Model calls", str(progress["calls_used"]))
    cost = progress["cost"]
    columns[2].metric("Provider cost",
                      f"${cost['actual_usd']:.4f}" if cost["reported_calls"]
                      else "not reported")
    columns[3].metric("Disagreements", str(progress["disagreements"]))

    chips = [
        theme_chip("sealed", flags.get("sealed")),
        theme_chip("classified", flags.get("classified")),
        theme_chip("evaluated", flags.get("evaluated")),
        theme_chip("explanations", progress["qualitative"]["started"]),
        theme_chip("report", progress["study_report"]),
    ]
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    status = last.get("status", "working")
    if advance is not None:
        if status == "busy":
            st.warning(last.get("detail")
                       or "Another session is running this study.")
            if st.button("Retry", type="primary"):
                st.session_state.pop("last_advance", None)
                st.rerun()
            return
        if status == "needs_pilot_review":
            _pilot_review(run_id, authorization, last)
        elif status in ("error", "blocked"):
            st.error(last.get("detail") or "The run needs attention.")
            if st.button("Retry", type="primary"):
                st.session_state.pop("last_advance", None)
                st.rerun()
            _controls(run_id, authorization)
        elif status == "stopped":
            st.warning(last.get("detail") or "Stopped.")
            _controls(run_id, authorization)
        elif status == "done":
            st.success("The analysis is complete.")
            _downloads(run_id)
        else:
            st.progress(
                counts["ok"] / counts["total"] if counts["total"] else 0.0,
                text=f"{counts['ok']} of {counts['total']} rows classified")
            _controls(run_id, authorization)
    else:
        st.info("Stopped. Rows already processed are saved.")
        _controls(run_id, authorization)
        _downloads(run_id)

    if last.get("log"):
        with st.expander("Latest engine output", expanded=False):
            st.code(last["log"], language=None)

    if advance is not None and status == "working" \
            and st.session_state.get("auto_run") and not stopped:
        steps = int(st.session_state.get("pipeline_steps") or 0) + 1
        st.session_state["pipeline_steps"] = steps
        if steps > 400:  # guard against any unforeseen non-progress loop
            st.session_state["auto_run"] = False
            st.error("The run stalled; press Resume or open the engine output.")
            return
        st.rerun()


def theme_chip(label: str, done: bool) -> str:
    from app import theme
    return theme.chip(f"{label} {'yes' if done else 'no'}",
                      "done" if done else "")


def _controls(run_id: str, authorization: dict) -> None:
    columns = st.columns(3)
    if st.session_state.get("stop_requested"):
        if columns[0].button("Resume", type="primary"):
            st.session_state["stop_requested"] = False
            st.session_state["auto_run"] = True
            st.rerun()
    else:
        if columns[0].button("Stop after this step"):
            st.session_state["stop_requested"] = True
            st.session_state["auto_run"] = False
            st.rerun()
    if columns[1].button("Re-authorize"):
        st.session_state["authorization"] = None
        st.session_state.pop("authorize_ack", None)
        st.rerun()


def _pilot_review(run_id: str, authorization: dict, last: dict) -> None:
    st.warning(last.get("detail") or "The pilot needs your decision.")
    rows = last.get("extra", {}).get("pilot_rows") or []
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    columns = st.columns(3)
    if columns[0].button("Continue anyway", type="primary"):
        authorization = dict(authorization)
        authorization["allow_unclean_pilot"] = True
        workspace.write_authorization(run_id, authorization)
        st.session_state["authorization"] = authorization
        st.session_state["auto_run"] = True
        st.rerun()
    if columns[1].button("Re-run the pilot"):
        pilot = workspace.run_dir(run_id) / "pilot"
        for name in ("pilot_predictions.csv", "approval.json"):
            path = pilot / name
            if path.exists():
                path.unlink()
        st.session_state["auto_run"] = True
        st.rerun()
    if columns[2].button("Stop"):
        st.session_state["stop_requested"] = True
        st.session_state["auto_run"] = False
        st.rerun()


def _downloads(run_id: str) -> None:
    directory = workspace.run_dir(run_id)
    study = report_support.study_report_paths(run_id)
    qualitative_pdf = report_support.qualitative_pdf_path(run_id)
    prediction_path = None
    for candidate in (directory / "results" / "frozen_predictions.csv",
                      directory / "predictions.csv"):
        if candidate.exists():
            prediction_path = candidate
            break

    columns = st.columns(2)
    if study["pdf"].exists():
        columns[0].download_button(
            "Download Full Report (PDF)", data=study["pdf"].read_bytes(),
            file_name=f"study_report_{run_id}.pdf", mime="application/pdf")
    if study["markdown"].exists():
        columns[1].download_button(
            "Full report (Markdown)", data=study["markdown"].read_bytes(),
            file_name=f"study_report_{run_id}.md", mime="text/markdown")
    if qualitative_pdf.exists():
        columns[0].download_button(
            "Qualitative report (PDF)", data=qualitative_pdf.read_bytes(),
            file_name=f"qualitative_report_{run_id}.pdf", mime="application/pdf")
    if prediction_path:
        columns[1].download_button(
            "Predictions (CSV)", data=prediction_path.read_bytes(),
            file_name=f"predictions_{run_id}.csv", mime="text/csv")
    files = result_view.result_files(directory)
    if files:
        columns[0].download_button(
            "Result tables (ZIP)",
            data=result_view.make_zip(directory, files),
            file_name=f"results_{run_id}.zip", mime="application/zip")
    provenance = result_view.provenance_files(directory)
    if provenance:
        columns[1].download_button(
            "Provenance bundle (ZIP)",
            data=result_view.make_zip(directory, provenance),
            file_name=f"provenance_{run_id}.zip", mime="application/zip")

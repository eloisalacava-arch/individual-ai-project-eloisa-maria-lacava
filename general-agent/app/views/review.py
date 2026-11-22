"""Step 2 (simple) - review the automatic proposal and correct only what matters."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from app import definitions, propose, registry, workspace
from app.state import go, p_check, p_number, p_radio, p_select, p_text

UNMAPPED = "(unmapped)"
EXCLUDE = "(exclude rows with this label)"


def _proposal_key(dataset: dict, benchmark: dict) -> str:
    raw = f"{dataset.get('sha256')}|{benchmark.get('sha256')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _apply(proposal: propose.Proposal) -> None:
    st.session_state.update({
        "dataset_id_column": proposal.dataset_id_column,
        "text_column": proposal.text_column,
        "benchmark_id_column": proposal.benchmark_id_column,
        "label_column": proposal.label_column,
        "classes": proposal.classes,
        "label_map": dict(proposal.label_map),
        "label_excluded": [],
        "make_id": False,
        "ordered": proposal.ordered,
        "provider_id": proposal.provider_id,
        "model_name": proposal.model_name,
        "rows_mode": proposal.rows_mode,
        "sample_n": proposal.sample_n or st.session_state.get("sample_n") or 100,
        "seed": proposal.seed,
        "max_calls": proposal.max_calls,
        "max_usd": proposal.max_usd,
        "qualitative_enabled": proposal.qualitative_enabled,
        "qualitative_sample_n": proposal.qualitative_sample_n,
        "qualitative_control_n": proposal.qualitative_control_n,
        "qualitative_max_calls": proposal.qualitative_max_calls,
        "qualitative_max_usd": proposal.qualitative_max_usd,
        "task_role": proposal.task_role,
        "decision_rule": proposal.decision_rule,
        "report_title": (proposal.report_title
                         or st.session_state.get("study_name") or ""),
    })


def _stats(dataset: dict, benchmark: dict, proposal: propose.Proposal) -> dict:
    from general_agent.discover import propose as discover_propose
    from general_agent.io import normalize_ids

    discovery = discover_propose(dataset["frame"], benchmark["frame"],
                                 propose.EMPTY_DATASET_CFG)
    stats = dict(discovery.stats or {})
    stats["label_values"] = list((proposal.label_map or {}).keys())
    label_column = proposal.label_column
    stats["duplicate_ids"] = stats.get("duplicate_ids") or {}
    if proposal.text_column and proposal.text_column in dataset["frame"].columns:
        stats["blank_text"] = int(
            dataset["frame"][proposal.text_column].astype("string").str.strip()
            .eq("").fillna(False).sum())
    else:
        stats["blank_text"] = 0
    return stats


def _blocking(stats: dict) -> list[str]:
    problems: list[str] = []
    if not st.session_state.get("make_id") and not st.session_state.get("dataset_id_column"):
        problems.append("Choose the observation ID column.")
    if not st.session_state.get("text_column"):
        problems.append("Choose the observation text column.")
    if not st.session_state.get("benchmark_id_column"):
        problems.append("Choose the benchmark ID column.")
    if not st.session_state.get("label_column"):
        problems.append("Choose the human label column.")
    classes = [c for c in st.session_state.get("classes", [])
               if c.get("name", "").strip() and c.get("definition", "").strip()]
    if len(classes) < 2:
        problems.append("Define at least two categories with a name and a definition.")
    placeholders = [c["name"] for c in classes
                    if definitions.is_placeholder_definition(c.get("definition", ""))]
    if placeholders:
        problems.append("Replace the placeholder definition(s) for "
                        + ", ".join(f"`{name}`" for name in placeholders)
                        + "; the definitions are shown to the AI in the prompt.")
    if stats.get("duplicate_ids"):
        problems.append(f"Duplicate IDs found: {stats['duplicate_ids']}. Remove or "
                        "merge those rows.")
    if stats.get("blank_text"):
        problems.append(f"{stats['blank_text']} observation(s) have empty text.")
    alignment = stats.get("id_alignment") or {}
    if (alignment.get("n_only_dataset") or alignment.get("n_only_benchmark")) \
            and st.session_state.get("on_id_mismatch") == "error":
        problems.append("IDs do not match. Choose 'keep only matching IDs' or fix "
                        "the files.")
    values = stats.get("label_values") or []
    mapping = st.session_state.get("label_map") or {}
    excluded = set(st.session_state.get("label_excluded") or [])
    missing = [value for value in values if value not in mapping and value not in excluded]
    if missing and st.session_state.get("on_unknown_label") == "error":
        problems.append(f"Map every human label value: {missing}.")
    present = set(mapping.values())
    declared_missing = [c["name"] for c in classes if c["name"] not in present]
    if declared_missing and not excluded and values:
        problems.append(f"These categories never appear in the mapped labels: "
                        f"{declared_missing}.")
    return problems


def render() -> None:
    st.subheader("2 · Review")
    st.caption("Everything below was proposed from your two files. Change only what "
               "is wrong; the highlighted items are the ones that need you.")

    dataset = st.session_state["dataset_meta"]
    benchmark = st.session_state["benchmark_meta"]
    key = _proposal_key(st.session_state.get("dataset_upload") or {},
                        st.session_state.get("benchmark_upload") or {})
    if st.session_state.get("proposal_key") != key:
        proposal = propose.build_proposal(dataset, benchmark,
                                          seed=st.session_state.get("seed", 20261116))
        _apply(proposal)
        st.session_state["proposal_key"] = key
        st.session_state["proposal_stats"] = _stats(dataset, benchmark, proposal)
        st.session_state["on_id_mismatch"] = "error"
        st.session_state["on_unknown_label"] = "error"

    stats = st.session_state.get("proposal_stats") or {}
    dataset_columns = dataset["columns"]
    benchmark_columns = benchmark["columns"]

    with st.container(border=True):
        st.markdown("#### Proposed plan")
        left, right = st.columns(2)
        with left:
            st.markdown("**Unmatched columns**")
            if st.session_state.get("make_id"):
                st.text_input("Observation ID", value="generated from row numbers",
                              disabled=True)
            else:
                p_select("Observation ID", dataset_columns, "dataset_id_column")
            p_select("Observation text", dataset_columns, "text_column")
        with right:
            st.markdown("**Benchmark columns**")
            p_select("Benchmark ID", benchmark_columns, "benchmark_id_column")
            p_select("Human label", benchmark_columns, "label_column")

    st.markdown("#### Categories and mapping")
    current = pd.DataFrame(st.session_state["classes"] or [],
                           columns=["name", "definition"])
    edited = st.data_editor(
        current, num_rows="dynamic", key="classes_editor", hide_index=True,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Category", required=True),
            "definition": st.column_config.TextColumn(
                "What it means", required=True, width="large",
                help="Shown to the AI in the prompt."),
        })
    rows = []
    for record in edited.to_dict(orient="records"):
        name = str(record.get("name") or "").strip()
        definition = str(record.get("definition") or "").strip()
        if name or definition:
            rows.append({"name": name, "definition": definition})
    st.session_state["classes"] = rows
    classes = [c["name"] for c in rows if c["name"]]

    values = stats.get("label_values") or []
    if values:
        mapping = dict(st.session_state.get("label_map") or {})
        excluded = list(st.session_state.get("label_excluded") or [])
        for value in values:
            key_name = "map_" + hashlib.md5(str(value).encode("utf-8")).hexdigest()[:10]
            options = [UNMAPPED] + classes + [EXCLUDE]
            if value in mapping and mapping[value] in classes:
                default = mapping[value]
            elif value in excluded:
                default = EXCLUDE
            else:
                default = UNMAPPED
            choice = st.selectbox(f"Human label {value!r}", options, key=key_name,
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

    with st.container(border=True):
        st.markdown("#### Model and budget")
        providers = registry.enabled_providers()
        if providers:
            provider_ids = [provider["id"] for provider in providers]
            provider_id = p_select(
                "Provider", provider_ids, "provider_id",
                format_func=lambda value: next(
                    provider["label"] for provider in providers
                    if provider["id"] == value))
            provider = registry.find_provider(provider_id)
            models = provider.get("models", [])
            model_names = [model["name"] for model in models]
            model_name = p_select(
                "Model", model_names, "model_name",
                format_func=lambda value: next(
                    model.get("label", model["name"]) for model in models
                    if model["name"] == value))
            model = registry.find_model(provider_id, model_name)
            key_env, present = registry.credential_status(model, provider)
            if key_env and not present:
                st.warning(f"Credential `{key_env}` is not set; validation and "
                           "sealing still work, classification needs it.")
            columns = st.columns(2)
            with columns[0]:
                p_number("Classification call cap", "max_calls", min_value=6,
                         max_value=registry.max_calls_ceiling(model), step=1)
            with columns[1]:
                p_number("Classification spend cap (USD)", "max_usd", min_value=0.0,
                         step=0.01, format="%.2f")
            p_check("Also explain disagreements with the AI when any are found "
                    "(hypotheses only)", "qualitative_enabled")

    blocking = _blocking(stats)
    alignment = stats.get("id_alignment") or {}
    if alignment.get("n_only_dataset") or alignment.get("n_only_benchmark"):
        st.markdown("#### Needs your decision")
        p_radio("IDs do not match exactly",
                options=["error", "inner_join"], key="on_id_mismatch",
                format_func=lambda value: {
                    "error": "Stop and let me fix the files",
                    "inner_join": f"Keep only the matching IDs "
                                  f"({alignment.get('n_only_dataset', 0)} / "
                                  f"{alignment.get('n_only_benchmark', 0)} dropped, "
                                  "counts printed)",
                }[value])
        blocking = _blocking(stats)

    if blocking:
        st.error("Before running:\n\n" + "\n".join(f"- {item}" for item in blocking))
    else:
        st.success("Everything needed is ready. Nothing else must be decided.")

    with st.expander("Advanced settings", expanded=False):
        p_check("Create the observation ID from row numbers", "make_id")
        p_radio("Rows to classify", options=["all", "subset"], key="rows_mode",
                horizontal=True,
                format_func=lambda value: {"all": "All rows",
                                           "subset": "A random subset"}[value])
        if st.session_state.get("rows_mode") == "subset":
            p_number("Rows in the subset", "sample_n", min_value=1, step=1)
        p_number("Random seed", "seed", min_value=1, step=1)
        p_check("Categories are ordered", "ordered")
        p_text("Task role (one sentence the AI sees)", "task_role")
        p_text("Decision rule (optional)", "decision_rule")
        p_text("Report title", "report_title")
        p_text("Report author", "report_author")
        columns = st.columns(2)
        with columns[0]:
            p_number("Disagreements to explain", "qualitative_sample_n", min_value=1,
                     step=1)
        with columns[1]:
            p_number("Control agreements to explain", "qualitative_control_n",
                     min_value=0, step=1)
        qualitative = bool(st.session_state.get("qualitative_enabled"))
        calls = int(st.session_state.get("max_calls") or 0)
        usd = float(st.session_state.get("max_usd") or 0)
        q_calls = int(st.session_state.get("qualitative_max_calls") or 0)
        q_usd = float(st.session_state.get("qualitative_max_usd") or 0)
        total_calls = calls + (q_calls if qualitative else 0)
        total_usd = usd + (q_usd if qualitative else 0)
        message = (
            f"Complete plan: validate, seal, pilot and classify at most {calls} calls "
            f"(${usd:.2f}). "
            + (f"Optional explanations may add at most {q_calls} calls "
               f"(${q_usd:.2f}) when disagreements exist. " if qualitative else "")
            + f"Combined maximum: {total_calls} calls and ${total_usd:.2f}.")
        st.caption(message)

    st.session_state["proposal"] = {"blocking": blocking}
    if st.button("Continue to analyze", type="primary", disabled=bool(blocking)):
        go(3)

"""Step 4 - provider, model, sample size, and cost limits."""

from __future__ import annotations

import streamlit as st

from app import registry, workspace
from app.config_builder import PILOT_N, suggest_max_calls
from app.state import go, p_check, p_number, p_radio, p_select


def render() -> None:
    st.subheader("4 · Model and limits")
    st.caption("Choose an approved provider. Credentials stay in the environment; "
               "this interface never asks for or shows a key.")

    providers = registry.enabled_providers()
    if not providers:
        st.error("No provider is enabled in configs/ui_providers.yaml.")
        return

    provider_ids = [provider["id"] for provider in providers]
    provider_id = p_select(
        "Provider", provider_ids, "provider_id",
        format_func=lambda value: next(
            provider["label"] for provider in providers if provider["id"] == value))
    provider = registry.find_provider(provider_id)
    st.caption(provider.get("access_note", ""))

    models = provider.get("models", [])
    model_names = [model["name"] for model in models]
    model_name = p_select(
        "Model", model_names, "model_name",
        format_func=lambda value: next(
            model.get("label", model["name"]) for model in models
            if model["name"] == value))
    model = registry.find_model(provider_id, model_name)

    key_env, present = registry.credential_status(model, provider)
    if key_env:
        if present:
            st.success(f"Credential available: environment variable {key_env} is set.")
        else:
            st.error(f"Credential missing: environment variable {key_env} is not set. "
                     "Export it in the shell that starts the interface, then reload.")
    st.caption("Temperature: not controlled by this provider. Recorded as "
               "`not_requested`.")
    st.code(model_name, language=None)

    st.markdown("#### Rows to classify")
    available = _available_rows()
    p_radio(
        "Sample",
        options=["all", "subset"],
        key="rows_mode",
        format_func=lambda value: {
            "all": f"All rows ({available})",
            "subset": "A random subset",
        }[value],
        horizontal=True)
    if st.session_state["rows_mode"] == "subset":
        p_number("Rows in the subset", "sample_n", min_value=1, step=1)
        st.caption("The draw is seeded and never stratified: the natural class "
                   "distribution is preserved.")
    p_number("Random seed", "seed", min_value=1, step=1)

    target = available if st.session_state["rows_mode"] == "all" else int(
        st.session_state.get("sample_n") or 1)
    ceiling = registry.max_calls_ceiling(model)
    suggested = min(suggest_max_calls(target), ceiling)

    st.markdown("#### Call and spend limits")
    st.caption(f"Every attempt is one call. Minimum needed: {PILOT_N} pilot + "
               f"{target} rows = {PILOT_N + target}. The cap also covers retries.")
    p_number("Maximum calls", "max_calls", min_value=PILOT_N + 1, max_value=ceiling,
             step=1)
    if st.session_state["max_calls"] < PILOT_N + target:
        st.warning(f"Raise the call limit to at least {PILOT_N + target} or the run "
                   "will stop before every row is classified.")
    price = registry.price_block(model)
    p_number("Maximum spend (USD)", "max_usd", min_value=0.0, step=0.01,
             format="%.2f")
    st.caption(f"Catalogue price: {price['input']} per million input tokens, "
               f"{price['output']} per million output tokens. The spend cap is a "
               "guardrail; the provider's reported cost is what is recorded. "
               f"Suggested call cap: {suggested}.")

    p_check(
        "I understand that the observation text will be sent to this provider.",
        "consent")

    ok = present and st.session_state["consent"]
    if st.button("Continue to validation", type="primary", disabled=not ok):
        go(5)
    if not present:
        st.caption("A missing credential blocks classification, not validation.")


def _available_rows() -> int:
    meta = st.session_state.get("dataset_meta")
    benchmark = st.session_state.get("benchmark_meta")
    rows = int(meta["n_rows"]) if meta else 0
    if benchmark is not None:
        rows = max(rows, 0)
    return rows or 1

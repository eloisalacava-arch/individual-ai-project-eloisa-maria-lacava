"""Focused checks on the shipped provider allow-list and its default.

The interface selects ``enabled_providers()[0]`` on a fresh session (see
``app/state.py``), so these tests pin the facts the README promises: an
OpenRouter ``openai_compatible`` entry is first and therefore pre-selected, the
local OpenCode CLI remains selectable, and a missing key produces clear,
non-fatal guidance. They never read a key value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import registry

pytest.importorskip("streamlit")


def test_openrouter_is_the_default_provider_on_a_fresh_install():
    providers = registry.enabled_providers()
    assert providers, "at least one provider must be enabled"
    default = providers[0]  # index 0 is the fresh-session default
    assert default["id"] == "openrouter"
    assert default["provider"] == "openai_compatible"
    assert default["base_url"] == "https://openrouter.ai/api/v1"
    assert default["api_key_env"] == "OPENROUTER_API_KEY"


def test_opencode_cli_remains_selectable():
    ids = [provider["id"] for provider in registry.enabled_providers()]
    assert "opencode" in ids
    opencode = registry.find_provider("opencode")
    assert opencode is not None
    assert opencode["provider"] == "opencode_cli"


def test_missing_openrouter_key_is_reported_without_reading_its_value(monkeypatch):
    provider = registry.find_provider("openrouter")
    model = registry.find_model("openrouter", provider["models"][0]["name"])

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    key_env, present = registry.credential_status(model, provider)
    assert key_env == "OPENROUTER_API_KEY"
    assert present is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-not-a-real-key")
    key_env, present = registry.credential_status(model, provider)
    assert key_env == "OPENROUTER_API_KEY"
    assert present is True


def test_openrouter_model_block_carries_endpoint_and_key_name_only():
    provider = registry.find_provider("openrouter")
    block = registry.model_block(provider, provider["models"][0])
    assert block["provider"] == "openai_compatible"
    assert block["base_url"] == "https://openrouter.ai/api/v1"
    assert block["api_key_env"] == "OPENROUTER_API_KEY"
    # The block carries the NAME of the variable, never a secret value.
    assert "sk-" not in str(block)


def test_missing_key_message_names_the_variable_and_the_export_command():
    message = registry.missing_key_message("OPENROUTER_API_KEY")
    assert "not set" in message.lower()
    assert "OPENROUTER_API_KEY" in message
    assert 'export OPENROUTER_API_KEY="your-key"' in message


def test_review_defaults_to_openrouter_and_warns_when_key_is_missing(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import workspace
    from general_agent.report import paths as report_paths
    from test_acceptance_new_user import _benchmark_bytes, _dataset_bytes

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    monkeypatch.setattr(report_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")

    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    dataset = workspace.stage_upload("s", "dataset", "observations.csv", _dataset_bytes())
    benchmark = workspace.stage_upload("s", "benchmark", "human_labels.csv", _benchmark_bytes())

    app = AppTest.from_file(str(app_path), default_timeout=60)
    app.session_state["dataset_upload"] = dataset
    app.session_state["benchmark_upload"] = benchmark
    app.session_state["dataset_meta"] = workspace.inspect_table(dataset["path"])
    app.session_state["benchmark_meta"] = workspace.inspect_table(benchmark["path"])
    app.run()
    next(field for field in app.text_input
         if field.key == "__w_study_name").set_value("provider_default")
    app.run()
    next(button for button in app.button if button.label == "Continue to review").click()
    app.run()

    # A fresh session pre-selects OpenRouter (index 0 of the allow-list) ...
    assert app.session_state["provider_id"] == "openrouter"
    assert app.session_state["model_name"] == "deepseek/deepseek-chat-v3.1"
    # ... and, with no key exported, explains how to set it.
    warnings = " ".join(element.value for element in app.warning)
    assert "OPENROUTER_API_KEY" in warnings
    assert "export OPENROUTER_API_KEY" in warnings

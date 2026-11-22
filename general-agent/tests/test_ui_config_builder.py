"""The UI builds a normal, fully explicit engine configuration."""

from __future__ import annotations

from pathlib import Path

from app import registry
from app.config_builder import suggest_max_calls, validate_spec
from general_agent.config import applied_defaults, load_config
from ui_helpers import make_ui_study


def test_generated_config_validates_and_states_every_field(tmp_path):
    study = make_ui_study(tmp_path)
    config_path = study["config_path"]
    cfg = load_config(config_path)
    assert applied_defaults(config_path) == {}
    assert Path(cfg["run"]["output_root"]) == (tmp_path / "runs").resolve()
    assert Path(cfg["dataset"]["path"]).exists()
    assert Path(cfg["benchmark"]["path"]).exists()
    assert cfg["dataset"]["id_column"] == "review_id"
    assert cfg["labels"]["mode"] == "single_label"


def test_ordered_classes_enable_weighted_kappa(tmp_path):
    study = make_ui_study(tmp_path, classes=3)
    cfg = load_config(study["config_path"])
    assert cfg["labels"]["ordered"] is True
    assert "quadratic" in cfg["evaluation"]["kappa"]


def test_binary_classes_use_unweighted_kappa_only(tmp_path):
    study = make_ui_study(tmp_path, classes=2)
    cfg = load_config(study["config_path"])
    assert cfg["labels"]["ordered"] is False
    assert cfg["evaluation"]["kappa"] == ["unweighted"]


def test_an_empty_label_map_is_omitted(tmp_path):
    study = make_ui_study(tmp_path, label_map={})
    cfg = load_config(study["config_path"])
    assert "label_map" not in cfg["benchmark"]


def test_a_numeric_label_map_is_written_verbatim(tmp_path):
    study = make_ui_study(tmp_path, label_map={"0": "negative", "1": "positive"})
    cfg = load_config(study["config_path"])
    assert cfg["benchmark"]["label_map"] == {"0": "negative", "1": "positive"}


def test_validate_spec_reports_human_readable_problems():
    problems = validate_spec({
        "classes": [{"name": "only one", "definition": "x"}],
        "dataset_id_column": None,
        "text_column": None,
        "benchmark_id_column": None,
        "label_column": None,
    })
    assert any("at least two" in problem for problem in problems)
    assert any("slug" in problem for problem in problems)
    assert any("text column" in problem for problem in problems)


def test_suggest_max_calls_covers_the_rows_and_a_retry_buffer():
    assert suggest_max_calls(0) == 10
    assert suggest_max_calls(100) == 115


def test_validate_spec_rejects_placeholder_definitions():
    from app.definitions import PLACEHOLDER_TEXT, is_placeholder_definition
    assert is_placeholder_definition(PLACEHOLDER_TEXT)
    assert is_placeholder_definition("")
    assert not is_placeholder_definition("an unfavourable overall stance")
    problems = validate_spec({
        "classes": [
            {"name": "positive", "definition": PLACEHOLDER_TEXT},
            {"name": "negative", "definition": "an unfavourable overall stance"},
        ],
        "dataset_id_column": "review_id",
        "text_column": "review_text",
        "benchmark_id_column": "review_id",
        "label_column": "sentiment",
    })
    assert any("placeholder" in problem.lower() for problem in problems)


def test_registry_exposes_a_provider_without_credentials():
    providers = registry.enabled_providers()
    assert providers
    provider = providers[0]
    assert provider["provider"] in {"opencode_cli", "openai_compatible"}
    model = provider["models"][0]
    block = registry.model_block(provider, model)
    assert block["name"] == model["name"]
    assert block["temperature"] is None
    if provider["provider"] == "opencode_cli":
        assert "api_key_env" not in block


def test_generated_config_never_holds_a_key(tmp_path):
    study = make_ui_study(tmp_path)
    text = study["config_path"].read_text(encoding="utf-8")
    assert "api_key" not in text.lower()

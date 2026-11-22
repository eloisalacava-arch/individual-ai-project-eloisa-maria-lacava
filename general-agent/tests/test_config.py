"""Config loading, schema validation, defaults, and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from general_agent.config import ConfigError, load_config
from helpers import FIXTURES, make_config


def test_template_config_is_valid_and_defaults_apply():
    template = Path(__file__).resolve().parents[1] / "configs" / "template.yaml"
    cfg = load_config(template)
    assert cfg["dataset"]["format"] == "auto"
    assert cfg["evaluation"]["bootstrap_draws"] == 2000
    assert cfg["evaluation"]["ci_level"] == 0.95
    assert cfg["model"]["max_attempts"] == 3
    assert cfg["benchmark"]["on_unknown_label"] == "error"
    assert cfg["labels"]["mode"] == "single_label"
    assert len(cfg["_meta"]["config_sha256"]) == 64


def test_output_root_from_the_template_points_at_the_agent_folder():
    template = Path(__file__).resolve().parents[1] / "configs" / "template.yaml"
    cfg = load_config(template)
    assert Path(cfg["run"]["output_root"]) == (
        template.parent.parent / "runs"
    ).resolve()


def test_output_root_default_resolves_next_to_the_config(tmp_path):
    path = make_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "output_root: " + str(tmp_path / "runs"), ""
    )
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    assert Path(cfg["run"]["output_root"]) == (tmp_path / "runs").resolve()


def test_absolute_paths_are_preserved(tmp_path):
    path = make_config(tmp_path)
    cfg = load_config(path)
    assert Path(cfg["dataset"]["path"]) == (FIXTURES / "dataset_clean.csv").resolve()
    assert Path(cfg["benchmark"]["path"]).is_absolute()


def test_relative_paths_resolve_against_the_config_directory(tmp_path):
    (tmp_path / "my_data.csv").write_text(
        "review_id,review_text\nr1,a short but sufficient observation text here\n",
        encoding="utf-8",
    )
    path = make_config(tmp_path, patch={"dataset": {"path": "my_data.csv"}})
    cfg = load_config(path)
    assert Path(cfg["dataset"]["path"]) == (tmp_path / "my_data.csv").resolve()


def test_unknown_key_is_rejected(tmp_path):
    path = make_config(tmp_path, patch={"run": {"typo_field": 1}})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "schema validation failed" in str(excinfo.value)


def test_missing_required_section_is_rejected(tmp_path):
    path = make_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("budget:", "budget_typo:")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_single_class_is_rejected(tmp_path):
    patch = {"labels": {"classes": [{"name": "positive", "definition": "only one"}]}}
    path = make_config(tmp_path, patch=patch)
    with pytest.raises(ConfigError):
        load_config(path)


def test_duplicate_class_names_are_rejected(tmp_path):
    patch = {"labels": {"classes": [
        {"name": "positive", "definition": "a"},
        {"name": "positive", "definition": "b"},
    ]}}
    path = make_config(tmp_path, patch=patch)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "duplicate" in str(excinfo.value)


def test_ordered_labels_require_weighted_kappa(tmp_path):
    path = make_config(tmp_path, patch={"labels": {"ordered": True}})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "quadratic" in str(excinfo.value)


def test_weighted_kappa_requires_ordered_labels(tmp_path):
    path = make_config(tmp_path, patch={
        "evaluation": {"kappa": ["unweighted", "quadratic"]}
    })
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "quadratic" in str(excinfo.value)


def test_label_map_target_must_be_a_declared_class(tmp_path):
    path = make_config(tmp_path, patch={
        "benchmark": {"label_map": {"0": "negatory", "1": "positive"}}
    })
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "negatory" in str(excinfo.value)


def test_primary_label_must_not_also_be_an_extra(tmp_path):
    path = make_config(tmp_path, patch={
        "benchmark": {"extra_label_columns": ["sentiment"]}
    })
    with pytest.raises(ConfigError):
        load_config(path)


def test_openai_compatible_requires_endpoint_and_key_name(tmp_path):
    path = make_config(tmp_path, patch={
        "model": {"provider": "openai_compatible", "name": "x/y"}
    })
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "base_url" in str(excinfo.value)


def test_openai_compatible_records_missing_key_without_failing(tmp_path, monkeypatch):
    monkeypatch.delenv("GA_TEST_KEY", raising=False)
    path = make_config(tmp_path, patch={
        "model": {
            "provider": "openai_compatible",
            "name": "x/y",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "GA_TEST_KEY",
        }
    })
    cfg = load_config(path)
    assert cfg["model"]["_api_key_env_present"] is False


def test_opencode_cli_rejects_key_settings(tmp_path):
    path = make_config(tmp_path, patch={"model": {"api_key_env": "SOME_KEY"}})
    with pytest.raises(ConfigError):
        load_config(path)
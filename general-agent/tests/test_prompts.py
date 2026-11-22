"""Prompt construction and the hash guard."""

from __future__ import annotations

import pytest

from general_agent.config import load_config
from general_agent.prompts import (
    PromptLeakError,
    build_prompt,
    prompt_template_sha,
    validate_rendering,
)
from helpers import make_config


@pytest.fixture
def cfg(tmp_path):
    return load_config(make_config(tmp_path))


def test_prompt_states_the_task_classes_and_contract(cfg):
    prompt = build_prompt(cfg, "A perfectly ordinary observation.")
    assert cfg["task"]["role"] in prompt
    for entry in cfg["labels"]["classes"]:
        assert entry["name"] in prompt
        assert entry["definition"] in prompt
    assert "ANSWER: <class>" in prompt
    assert "positive, negative" in prompt
    assert "Do not use outside knowledge" in prompt


def test_prompt_contains_the_observation_in_the_delimited_block(cfg):
    text = "The battery lasts two days, but the screen scratches easily."
    prompt = build_prompt(cfg, text)
    start = prompt.index("<<<\n") + len("<<<\n")
    end = prompt.index("\n>>>", start)
    assert prompt[start:end] == text


def test_template_hash_is_stable_across_observations(cfg):
    first = build_prompt(cfg, "one observation")
    second = build_prompt(cfg, "a completely different observation")
    assert first != second
    assert prompt_template_sha(cfg) == prompt_template_sha(cfg)


def test_template_hash_changes_when_a_class_definition_changes(tmp_path):
    first = load_config(make_config(tmp_path))
    changed = {
        "labels": {"classes": [
            {"name": "positive", "definition": "a different favourable definition"},
            {"name": "negative", "definition": "unfavourable overall stance"},
        ]}
    }
    second = load_config(make_config(tmp_path, patch=changed, name="changed.yaml"))
    assert prompt_template_sha(first) != prompt_template_sha(second)


def test_validate_rendering_accepts_the_exact_prompt(cfg):
    text = "A short observation about a product."
    validate_rendering(build_prompt(cfg, text), cfg, text)


def test_validate_rendering_rejects_tampering(cfg):
    text = "A short observation about a product."
    tampered = build_prompt(cfg, text).replace("Do not explain", "Explain fully")
    with pytest.raises(PromptLeakError):
        validate_rendering(tampered, cfg, text)


def test_validate_rendering_rejects_a_different_observation(cfg):
    prompt = build_prompt(cfg, "first observation")
    with pytest.raises(PromptLeakError):
        validate_rendering(prompt, cfg, "second observation")


def test_observation_that_equals_a_class_name_is_not_flagged(cfg):
    # A naive `text in prompt` count would fail here; the delimiter check must not.
    text = "positive"
    prompt = build_prompt(cfg, text)
    validate_rendering(prompt, cfg, text)


def test_empty_text_is_rejected(cfg):
    with pytest.raises(ValueError):
        build_prompt(cfg, "   ")


def test_prompt_contains_no_counts_or_distribution(cfg):
    prompt = build_prompt(cfg, "An observation.")
    for fragment in ("%", "distribution", "accuracy", "prevalence"):
        assert fragment not in prompt.lower()

def test_observation_containing_delimiters_still_validates(cfg):
    text = "Line one with a <<< marker and a >>> marker, then more text."
    prompt = build_prompt(cfg, text)
    validate_rendering(prompt, cfg, text)

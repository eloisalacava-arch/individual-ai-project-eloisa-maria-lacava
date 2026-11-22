"""The explanation prompt is deterministic and carries one row only."""

from __future__ import annotations

from general_agent.qualitative import prompts
from qualitative_helpers import evaluated_run, qualitative_settings

MARKER_OTHER = "OTHER-ROW-POISON-7c1a"


def _classes():
    return [
        {"name": "negative", "definition": "unfavourable"},
        {"name": "positive", "definition": "favourable"},
    ]


def test_prompt_contains_classes_both_labels_and_the_observation():
    settings = {"vocabulary_version": 1}
    item = {"group": "disagreement", "reference": "negative",
            "predicted": "positive", "text": "The film was not bad at all."}
    prompt = prompts.build_prompt(settings, _classes(), item)
    prompts.validate_rendering(prompt, item["text"])
    assert "negative: unfavourable" in prompt
    assert "Human label: negative" in prompt
    assert "Classifier answer: positive" in prompt
    assert "hypotheses only" in prompt.lower()


def test_the_two_variants_have_different_template_hashes():
    settings = {"vocabulary_version": 1}
    hashes = prompts.prompt_template_hashes(settings, _classes())
    assert hashes["disagreement"] != hashes["agreement"]
    assert set(hashes) == {"disagreement", "agreement"}


def test_a_prompt_contains_no_other_row():
    settings = {"vocabulary_version": 1}
    item = {"group": "disagreement", "reference": "negative",
            "predicted": "positive", "text": "A short observation."}
    prompt = prompts.build_prompt(settings, _classes(), item)
    assert MARKER_OTHER not in prompt


def test_template_hash_is_stable_across_observations(tmp_path):
    study = evaluated_run(tmp_path)
    settings = qualitative_settings(study)
    classes = settings["_meta"]["classes"]
    first = prompts.prompt_template_sha(settings, classes, "disagreement")
    second = prompts.prompt_template_sha(settings, classes, "disagreement")
    assert first == second

"""Deterministic explanation prompts and the render guard.

The prompt is a pure function of the settings, the category definitions, and one
observation. The hash of that function (with the observation and labels held
constant) is recorded with every attempt and bound into the approval file, so a
changed category definition or vocabulary forces a new pilot.
"""

from __future__ import annotations

from ..provenance import sha256_text
from . import schema

PLACEHOLDER_TEXT = "<OBSERVATION>"
PLACEHOLDER_HUMAN = "<HUMAN_LABEL>"
PLACEHOLDER_AI = "<AI_LABEL>"


class PromptLeakError(RuntimeError):
    """Raised when a rendered prompt is not the deterministic template."""


DISAGREEMENT_HEADER = """You are assisting a methodological review of a classification study.

A classifier was given only the observation text and the category definitions
below. It did not see the human label. Here the classifier answer is NOT the
human label.

Propose *possible* reasons the two labels differ. These are hypotheses only:
do not state which label is correct, do not claim the classifier or the human
made a mistake, and do not assert any cause.

Categories:
{classes_block}

Observation:
<<<
{text}
>>>

Human label: {human}
Classifier answer: {ai}

Allowed reason codes:
{reasons_block}

Reply with exactly one JSON object and no other text:
{contract}
"""


AGREEMENT_HEADER = """You are assisting a methodological review of a classification study.

A classifier was given only the observation text and the category definitions
below. It did not see the human label. Here the classifier answer agrees with
the human label.

Describe which features of the text could support that shared reading, and
whether any tension remains that another careful reader might weigh differently.
These are hypotheses only: do not assert a cause and do not claim any label is
proven correct.

Categories:
{classes_block}

Observation:
<<<
{text}
>>>

Human label: {human}
Classifier answer: {ai}

Allowed reason codes:
{reasons_block}

Reply with exactly one JSON object and no other text:
{contract}
"""


def variant_for(group: str) -> str:
    return "agreement" if group == "agreement" else "disagreement"


def _classes_block(classes: list[dict]) -> str:
    return "\n".join(f"- {entry['name']}: {entry['definition']}"
                     for entry in classes)


def _render(settings: dict, classes: list[dict], variant: str, text: str,
            human: str, ai: str) -> str:
    header = AGREEMENT_HEADER if variant == "agreement" else DISAGREEMENT_HEADER
    return header.format(
        classes_block=_classes_block(classes),
        text=text,
        human=human,
        ai=ai,
        reasons_block=schema.reasons_block(
            int(settings.get("vocabulary_version", 1))),
        contract=schema.json_contract(
            int(settings.get("vocabulary_version", 1))),
    )


def build_prompt(settings: dict, classes: list[dict], item: dict) -> str:
    text = item["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("observation text must be a non-empty string")
    return _render(settings, classes, variant_for(item["group"]),
                   text, str(item["reference"]), str(item["predicted"]))


def prompt_template_sha(settings: dict, classes: list[dict],
                        variant: str) -> str:
    return sha256_text(_render(settings, classes, variant, PLACEHOLDER_TEXT,
                               PLACEHOLDER_HUMAN, PLACEHOLDER_AI))


def prompt_template_hashes(settings: dict, classes: list[dict]) -> dict:
    return {
        "disagreement": prompt_template_sha(settings, classes, "disagreement"),
        "agreement": prompt_template_sha(settings, classes, "agreement"),
    }


def validate_rendering(prompt: str, text: str) -> None:
    """Fail closed if the delimited observation is not the text verbatim."""
    opener = "Observation:\n<<<\n"
    closer = "\n>>>\n\nHuman label:"
    if prompt.count(opener) != 1 or closer not in prompt:
        raise PromptLeakError("observation delimiters are missing from the prompt")
    start = prompt.index(opener) + len(opener)
    end = prompt.rindex(closer)
    if prompt[start:end] != text:
        raise PromptLeakError("the delimited observation is not the row text verbatim")

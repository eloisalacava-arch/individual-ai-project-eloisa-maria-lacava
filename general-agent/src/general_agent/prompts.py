"""Prompt construction and the hash guard.

The prompt is a pure function of the config and one observation string. That
makes it testable and makes leakage detectable: a prompt that does not equal
``build_prompt(cfg, text)`` exactly is rejected before the call is made.

The prompt contains no label distribution, no counts, and no example taken from
the dataset. Only the declared class names and their definitions appear, because
the task cannot be stated without them.
"""

from __future__ import annotations

from .provenance import sha256_text

PLACEHOLDER = "<OBSERVATION>"

HEADER = """You are {role}.

Classify the observation below into exactly one of these classes:
{classes_block}

{rules}
Do not explain your answer.

Observation:
<<<
{text}
>>>

Reply with exactly one line:
{contract}
where <class> is one of: {class_list}
"""


class PromptLeakError(RuntimeError):
    """Raised when a rendered prompt does not match the deterministic template."""


def _classes_block(cfg: dict) -> str:
    return "\n".join(
        f"- {entry['name']}: {entry['definition']}"
        for entry in cfg["labels"]["classes"]
    )


def _rules(cfg: dict) -> str:
    rules: list[str] = []
    decision_rule = cfg["task"].get("decision_rule")
    if decision_rule:
        rules.append(f"- {decision_rule.strip()}")
    if cfg["task"].get("forbid_outside_knowledge"):
        rules.append("- Do not use outside knowledge; judge only the text.")
    return "\n".join(rules) if rules else "- Judge only the observation text."


def _contract(cfg: dict) -> str:
    return cfg["task"]["output_contract"].replace("{class}", "<class>")


def build_prompt(cfg: dict, text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("observation text must be a string")
    if not text.strip():
        raise ValueError("observation text is empty")
    return HEADER.format(
        role=cfg["task"]["role"],
        classes_block=_classes_block(cfg),
        rules=_rules(cfg),
        text=text,
        contract=_contract(cfg),
        class_list=", ".join(entry["name"] for entry in cfg["labels"]["classes"]),
    )


def prompt_template_sha(cfg: dict) -> str:
    """Hash of the template with the observation held constant.

    Stable across rows, and changes if the task, class definitions, rule, or
    output contract change. Recorded with every attempt and in the approval file.
    """
    return sha256_text(build_prompt(cfg, PLACEHOLDER))


def prompt_sha(cfg: dict, text: str) -> str:
    return sha256_text(build_prompt(cfg, text))


def validate_rendering(prompt: str, cfg: dict, text: str) -> None:
    """Fail closed if the prompt is not exactly the deterministic rendering."""
    expected = build_prompt(cfg, text)
    if prompt != expected:
        raise PromptLeakError(
            "rendered prompt differs from build_prompt(cfg, text); refusing to send"
        )
    opener = "Observation:\n<<<\n"
    closer = "\n>>>\n\nReply with exactly one line:"
    if prompt.count(opener) < 1 or closer not in prompt:
        raise PromptLeakError("observation delimiters are missing from the prompt")
    start = prompt.index(opener) + len(opener)
    # rindex: the template's closer is the last one, even if the observation
    # itself contains delimiter-like text.
    end = prompt.rindex(closer)
    if prompt[start:end] != text:
        raise PromptLeakError("the delimited observation is not the configured text verbatim")
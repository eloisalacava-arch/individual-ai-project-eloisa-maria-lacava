"""The explanation contract, the controlled vocabulary, and a strict parser.

The vocabulary is a deliberate construct, versioned so that a change forces a
new pilot: patterns are only comparable within one version. Unknown reason
values are preserved raw and mapped to ``other``; nothing is silently dropped.
"""

from __future__ import annotations

import json
import re

CURRENT_VOCABULARY_VERSION = 1
CURRENT_PROMPT_VERSION = 1

OTHER = "other"

VOCABULARY_V1: dict[str, str] = {
    "mixed_or_conflicting_signals":
        "parts of the observation point in different directions",
    "figurative_or_sarcastic_language":
        "irony, sarcasm, or metaphor could invert the literal reading",
    "category_definition_overlap":
        "the declared categories are close enough to be confused here",
    "insufficient_context_in_observation":
        "the text alone may not carry the information a reader needs",
    "domain_or_world_knowledge":
        "resolving it needs facts outside the observation",
    "subtle_or_low_intensity_sentiment":
        "the stance is weak, hedged, or nearly neutral",
    "text_noise_or_truncation":
        "formatting, encoding, or a cut-off may distort the text",
    "reference_label_ambiguity":
        "the human label itself may be debatable",
    "annotation_error_possible":
        "a labelling mistake is possible (hypothesis only)",
    OTHER:
        "none of the above fits well; explain in the summary",
}

VOCABULARIES = {1: VOCABULARY_V1}

# Readable labels for reports. The CSV tables keep the machine codes; only the
# rendered Markdown and PDF show these labels.
REASON_LABELS: dict[str, str] = {
    "mixed_or_conflicting_signals": "Mixed or conflicting signals",
    "figurative_or_sarcastic_language": "Figurative or sarcastic language",
    "category_definition_overlap": "Overlapping category definitions",
    "insufficient_context_in_observation": "Insufficient context in the observation",
    "domain_or_world_knowledge": "Needs domain or world knowledge",
    "subtle_or_low_intensity_sentiment": "Subtle or low-intensity sentiment",
    "text_noise_or_truncation": "Text noise or truncation",
    "reference_label_ambiguity": "Ambiguous reference label",
    "annotation_error_possible": "Possible annotation error",
    OTHER: "Other",
}


def reason_label(code: str) -> str:
    """A human-readable label for a reason code, falling back to the code."""
    return REASON_LABELS.get(code, code)


def reason_label_lower(code: str) -> str:
    """The label with a lowercase first letter, for use mid-sentence."""
    label = reason_label(code)
    return label[:1].lower() + label[1:] if label else label


class ExplanationParseError(RuntimeError):
    """Raised only for programming errors; malformed replies use statuses."""


def vocabulary(version: int = CURRENT_VOCABULARY_VERSION) -> dict[str, str]:
    if version not in VOCABULARIES:
        raise ExplanationParseError(f"unknown vocabulary version {version!r}")
    return VOCABULARIES[version]


def reason_codes(version: int = CURRENT_VOCABULARY_VERSION) -> list[str]:
    return list(vocabulary(version).keys())


def _strip_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z0-9_]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def _extract_json(text: str) -> dict | None:
    candidate = _strip_fences(text or "")
    if not candidate:
        return None
    try:
        loaded = json.loads(candidate)
        return loaded if isinstance(loaded, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        loaded = json.loads(candidate[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _normalise(text: str) -> str:
    return " ".join(str(text).split())


def _as_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_explanation(text: str, allowed_reasons: list[str],
                      observation: str) -> tuple[str, dict | None, str]:
    """Return (status, explanation, detail).

    status: ok | parse_fail | parse_missing_summary
    A missing or unusable summary is a failure; an unparseable reply is never
    turned into a fabricated explanation.
    """
    payload = _extract_json(text or "")
    if payload is None:
        return "parse_fail", None, "no JSON object in the reply"

    summary = _as_text(payload.get("summary"))
    if not summary:
        return "parse_missing_summary", None, "the reply has no usable summary"

    raw_reasons = payload.get("reasons")
    if isinstance(raw_reasons, str):
        raw_reasons = [raw_reasons]
    if not isinstance(raw_reasons, list):
        raw_reasons = []
    reasons_raw: list[str] = []
    reasons: list[str] = []
    for item in raw_reasons:
        value = _as_text(item)
        if not value:
            continue
        reasons_raw.append(value)
        reasons.append(value if value in allowed_reasons else OTHER)
    if not reasons:
        reasons = [OTHER]
    # deduplicate but keep order, cap at three
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    reasons = seen[:3]

    evidence_raw = _as_text(payload.get("evidence"))
    evidence = ""
    evidence_valid = False
    if evidence_raw:
        if _normalise(evidence_raw) in _normalise(observation):
            evidence = evidence_raw
            evidence_valid = True

    resolve = _as_text(payload.get("what_would_resolve"))
    confidence = _as_text(payload.get("confidence")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    return "ok", {
        "summary": summary[:1000],
        "reasons": reasons,
        "reasons_raw": reasons_raw,
        "evidence": evidence,
        "evidence_valid": evidence_valid,
        "what_would_resolve": resolve[:500],
        "confidence": confidence,
    }, f"{len(reasons)} reason(s)"


def json_contract(version: int = CURRENT_VOCABULARY_VERSION) -> str:
    return (
        "{\n"
        '  "summary": "one or two sentences; a hypothesis only",\n'
        '  "reasons": ["one to three codes from the allowed list"],\n'
        '  "evidence": "a short span copied verbatim from the observation, '
        'or \\"\\"",\n'
        '  "what_would_resolve": "one sentence, e.g. more context or '
        'per-annotator labels",\n'
        '  "confidence": "low | medium | high"\n'
        "}"
    )


def reasons_block(version: int = CURRENT_VOCABULARY_VERSION) -> str:
    return "\n".join(f"- {code}: {gloss}"
                     for code, gloss in vocabulary(version).items())

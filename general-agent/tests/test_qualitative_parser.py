"""The explanation parser never invents an explanation."""

from __future__ import annotations

from general_agent.qualitative.schema import (
    OTHER, parse_explanation, reason_codes,
)

REASONS = reason_codes(1)
OBSERVATION = "The plot dragged, but the ending was genuinely moving."


def test_a_valid_reply_is_parsed():
    raw = ('{"summary": "Mixed signals.", "reasons": ["mixed_or_conflicting_signals"],'
           ' "evidence": "but the ending was genuinely moving",'
           ' "what_would_resolve": "per-annotator labels", "confidence": "high"}')
    status, payload, detail = parse_explanation(raw, REASONS, OBSERVATION)
    assert status == "ok"
    assert payload["reasons"] == ["mixed_or_conflicting_signals"]
    assert payload["evidence"] == "but the ending was genuinely moving"
    assert payload["evidence_valid"] is True
    assert payload["confidence"] == "high"


def test_a_fenced_reply_is_parsed():
    raw = "```json\n{\"summary\": \"x\", \"reasons\": [\"other\"], \"confidence\": \"low\"}\n```"
    status, payload, _ = parse_explanation(raw, REASONS, OBSERVATION)
    assert status == "ok"
    assert payload["reasons"] == [OTHER]


def test_prose_without_json_fails():
    status, payload, detail = parse_explanation("I think the model was wrong.", REASONS,
                                                OBSERVATION)
    assert status == "parse_fail"
    assert payload is None


def test_missing_summary_fails():
    status, payload, _ = parse_explanation('{"reasons": ["other"]}', REASONS, OBSERVATION)
    assert status == "parse_missing_summary"
    assert payload is None


def test_unknown_reason_is_preserved_raw_and_mapped_to_other():
    raw = ('{"summary": "x", "reasons": ["totally_made_up"], "confidence": "low"}')
    status, payload, _ = parse_explanation(raw, REASONS, OBSERVATION)
    assert status == "ok"
    assert payload["reasons"] == [OTHER]
    assert payload["reasons_raw"] == ["totally_made_up"]


def test_non_verbatim_evidence_is_blanked_and_flagged():
    raw = ('{"summary": "x", "reasons": ["other"], "evidence": "not in the text",'
           ' "confidence": "medium"}')
    status, payload, _ = parse_explanation(raw, REASONS, OBSERVATION)
    assert status == "ok"
    assert payload["evidence"] == ""
    assert payload["evidence_valid"] is False


def test_reasons_are_capped_at_three_and_deduplicated():
    raw = ('{"summary": "x", "reasons": ["other", "other", "mixed_or_conflicting_signals",'
           ' "text_noise_or_truncation", "figurative_or_sarcastic_language"],'
           ' "confidence": "medium"}')
    status, payload, _ = parse_explanation(raw, REASONS, OBSERVATION)
    assert status == "ok"
    assert payload["reasons"] == [
        "other", "mixed_or_conflicting_signals", "text_noise_or_truncation"]


def test_bad_confidence_defaults_to_low():
    raw = '{"summary": "x", "reasons": ["other"], "confidence": "certain"}'
    status, payload, _ = parse_explanation(raw, REASONS, OBSERVATION)
    assert payload["confidence"] == "low"


def test_missing_reasons_fall_back_to_other():
    raw = '{"summary": "x", "confidence": "medium"}'
    status, payload, _ = parse_explanation(raw, REASONS, OBSERVATION)
    assert status == "ok"
    assert payload["reasons"] == [OTHER]

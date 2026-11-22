"""Answer parsing: strict contract, no guessing."""

from __future__ import annotations

from general_agent.parsing import extract_label, is_transient

CLASSES = ["positive", "negative"]


def test_simple_answer_is_parsed():
    status, label, _ = extract_label("ANSWER: positive", CLASSES)
    assert (status, label) == ("ok", "positive")


def test_case_and_whitespace_are_tolerated():
    status, label, _ = extract_label("  answer:   Negative  ", CLASSES)
    assert (status, label) == ("ok", "negative")


def test_trailing_period_and_quotes_are_ignored():
    assert extract_label('ANSWER: "positive".', CLASSES)[:2] == ("ok", "positive")


def test_last_answer_wins_when_they_agree():
    status, label, detail = extract_label(
        "ANSWER: positive\nsome prose\nANSWER: positive", CLASSES
    )
    assert (status, label) == ("ok", "positive")
    assert "2 ANSWER" in detail


def test_conflicting_answers_are_ambiguous():
    status, label, detail = extract_label(
        "ANSWER: positive\nANSWER: negative", CLASSES
    )
    assert status == "parse_ambiguous"
    assert label is None
    assert "positive" in detail and "negative" in detail


def test_unknown_class_is_not_guessed():
    status, label, _ = extract_label("ANSWER: mixed", CLASSES)
    assert status == "parse_unknown_class"
    assert label is None


def test_missing_answer_line_fails():
    status, label, _ = extract_label("I think it is probably positive.", CLASSES)
    assert status == "parse_fail"
    assert label is None


def test_empty_reply_fails():
    assert extract_label("", CLASSES)[0] == "parse_fail"
    assert extract_label(None, CLASSES)[0] == "parse_fail"


def test_answer_inside_prose_is_not_parsed():
    status, _, _ = extract_label("The line ANSWER: positive is required.", CLASSES)
    assert status == "parse_fail"


def test_all_parse_problems_are_transient():
    for status in ("parse_fail", "parse_ambiguous", "parse_unknown_class"):
        assert is_transient(status)


def test_blocked_statuses_are_not_transient():
    for status in ("blocked_auth", "blocked_billing", "blocked_rate_limit"):
        assert not is_transient(status)
"""De-duplicate methodological limitations in generated reports.

The per-study report and the comparison report both merge a generic set of
caveats with study-specific ones (and, when a study has a qualitative run, the
qualitative caveats). The generic and specific statements frequently restate the
same point, which reads as repetition. This helper keeps each distinct caveat
once, clause by clause, while preserving any study-specific detail the generic
text does not already cover.

It is deliberately conservative: a clause is dropped only when every one of its
content words is already covered by an earlier clause, after a small synonym
map. Distinct caveats are therefore never removed by accident.
"""

from __future__ import annotations

import re

# Words that carry no caveat content on their own.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "of", "in", "on", "to", "for", "with", "this", "that",
    "these", "those", "it", "its", "as", "by", "from", "may", "can",
    "could", "does", "do", "but", "if", "at", "within", "also", "than",
    "then", "so", "such", "one", "all",
}

# Phrase-level synonyms so that reworded caveats collapse onto one form.
_SYNONYMS = (
    (re.compile(r"\bhuman labels?\b", re.I), "reference"),
    (re.compile(r"\bnot adjudicated truth\b", re.I),
     "not an adjudicated gold standard"),
    (re.compile(r"\bnot truth\b", re.I), "not a gold standard"),
    (re.compile(r"\bmodel temperature\b", re.I), "temperature"),
    (re.compile(r"\brepeated runs may differ\b", re.I), "model versions drift"),
    (re.compile(r"\bevery interval\b", re.I), "intervals"),
    (re.compile(r"\binterval\b", re.I), "intervals"),
)

# Redundant tails that repeat an already-stated caveat inside a longer clause.
_REDUNDANT_PHRASES = (
    re.compile(r",?\s*not an adjudicated gold standard\b", re.I),
    re.compile(r",?\s*not adjudicated truth\b", re.I),
    re.compile(r",?\s*not a gold standard\b", re.I),
    re.compile(r",?\s*not truth\b", re.I),
)

# A bare row count is already reported in the metadata table and the executive
# summary; repeating it as a limitation adds nothing.
_ROW_ONLY = re.compile(r"^\d[\d,]*\s+analysed rows$", re.I)

_PREFIX = re.compile(r"^`[^`]+`:\s*")
_CLAUSE_SPLIT = re.compile(r"\s*;\s*|\s*\.\s+")
_TERMINALS = ".!?"


def _normalise(text: str) -> frozenset[str]:
    lowered = text.lower()
    for pattern, replacement in _SYNONYMS:
        lowered = pattern.sub(replacement, lowered)
    return frozenset(
        token for token in re.findall(r"[a-z0-9]+", lowered)
        if token not in _STOPWORDS
    )


def _is_covered(tokens: frozenset[str], seen: list[frozenset[str]]) -> bool:
    return any(tokens <= previous for previous in seen)


def _clauses(text: str) -> list[str]:
    return [part.strip(" ,;.") for part in _CLAUSE_SPLIT.split(text.strip())]


def dedupe_limitations(lines: list[str],
                       prefix: re.Pattern[str] | None = None) -> list[str]:
    """Return the limitations as one bullet per distinct caveat.

    ``lines`` are the raw limitation strings (a leading ``- `` is optional).
    ``prefix`` optionally matches a leading ``run_id`` label (as used by the
    comparison report); it is stripped before comparison and restored verbatim on
    a line that still has unique content. Emitted lines are always ``- `` bullets.
    """
    seen: list[frozenset[str]] = []
    output: list[str] = []
    for raw in lines:
        text = raw.strip()
        label = ""
        if text.startswith("- "):
            text = text[2:].strip()
        if prefix is not None:
            match = prefix.match(text)
            if match:
                label = match.group(0)
                text = text[len(label):].strip()
        kept: list[str] = []
        modified = False
        for clause in _clauses(text):
            cleaned = clause
            for pattern in _REDUNDANT_PHRASES:
                match = pattern.search(cleaned)
                if match and _is_covered(_normalise(match.group(0)), seen):
                    cleaned = (cleaned[:match.start()] + cleaned[match.end():]
                               ).strip(" ,;.")
                    modified = True
            cleaned = cleaned.strip(" ,;.")
            if not cleaned or _ROW_ONLY.match(cleaned):
                modified = True
                continue
            tokens = _normalise(cleaned)
            if not tokens or _is_covered(tokens, seen):
                modified = True
                continue
            kept.append(cleaned)
            seen.append(tokens)
        if not modified:
            output.append(f"- {label}{text}".rstrip())
        elif kept:
            joined = "; ".join([kept[0][0].upper() + kept[0][1:]] + kept[1:])
            output.append(f"- {label}{joined.rstrip(_TERMINALS)}.")
    return output

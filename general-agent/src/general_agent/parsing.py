"""Answer parsing.

Status vocabulary (stored verbatim in the predictions file):

  ok                  exactly one declared class was answered; last answer wins
  parse_fail          no ANSWER line
  parse_ambiguous     conflicting declared classes in the same reply
  parse_unknown_class an ANSWER line naming something outside the class set

Nothing is imputed. A non-ok row keeps an empty label and carries its status.
"""

from __future__ import annotations

import re

ANSWER_LINE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _canonical_map(classes: list[str]) -> dict[str, str]:
    return {name.strip().casefold(): name for name in classes}


def _normalise(candidate: str) -> str:
    value = candidate.strip()
    value = value.strip("\"'`").strip()
    value = value.rstrip(".").strip()
    value = value.strip("\"'`").strip()
    return value


def extract_label(text: str | None, classes: list[str]) -> tuple[str, str | None, str]:
    """Return (status, label_or_None, detail)."""
    if not text or not text.strip():
        return "parse_fail", None, "empty reply"

    matches = [_normalise(match) for match in ANSWER_LINE.findall(text)]
    if not matches:
        return "parse_fail", None, "no ANSWER line"

    canonical = _canonical_map(classes)
    resolved: list[str] = []
    unknown: list[str] = []
    for match in matches:
        target = canonical.get(match.casefold())
        if target is None:
            unknown.append(match)
        else:
            resolved.append(target)

    if unknown:
        return (
            "parse_unknown_class",
            None,
            f"answered outside the declared classes: {sorted(set(unknown))}",
        )

    distinct = sorted(set(resolved))
    if len(distinct) > 1:
        return "parse_ambiguous", None, f"conflicting answers in one reply: {distinct}"

    return "ok", distinct[0], f"{len(matches)} ANSWER line(s)"


def is_transient(status: str) -> bool:
    """Retry on transport/parse problems; never retry a fatal provider block."""
    return status in {
        "api_error", "timeout", "network_error", "local_error",
        "parse_fail", "parse_ambiguous", "parse_unknown_class", "empty_reply",
    }
"""Validation helpers for category definitions in the interface."""

from __future__ import annotations

import re

# The exact text proposed when a category definition still needs to be written.
PLACEHOLDER_TEXT = "Replace this with what the category means."

_PLACEHOLDER = re.compile(
    r"\bplaceholder\b|" + re.escape("replace this with what the category means"),
    re.I,
)


def is_placeholder_definition(text: str) -> bool:
    """True when a definition is empty or still the automatic placeholder."""
    value = " ".join(str(text or "").split())
    if not value:
        return True
    return bool(_PLACEHOLDER.search(value))

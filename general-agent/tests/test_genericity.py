"""The evaluation layer must contain no dataset-specific code."""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "general_agent"
FORBIDDEN = ("imdb", "phrasebank", "financial", "movie review", "sentence_75")


@pytest.mark.parametrize("path", sorted((SRC / "evaluate").glob("*.py")))
def test_no_dataset_name_in_evaluation_code(path: Path):
    text = path.read_text(encoding="utf-8").lower()
    hits = [term for term in FORBIDDEN if term in text]
    assert not hits, f"{path.name} contains dataset-specific terms: {hits}"


@pytest.mark.parametrize("path", sorted((SRC / "qualitative").glob("*.py")))
def test_no_dataset_name_in_qualitative_code(path: Path):
    text = path.read_text(encoding="utf-8").lower()
    hits = [term for term in FORBIDDEN if term in text]
    assert not hits, f"{path.name} contains dataset-specific terms: {hits}"


@pytest.mark.parametrize("path", sorted((SRC / "report").glob("*.py")))
def test_no_dataset_name_in_report_code(path: Path):
    text = path.read_text(encoding="utf-8").lower()
    hits = [term for term in FORBIDDEN if term in text]
    assert not hits, f"{path.name} contains dataset-specific terms: {hits}"

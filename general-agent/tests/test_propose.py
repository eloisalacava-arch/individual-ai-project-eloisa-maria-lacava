"""P1: the automatic proposal engine."""

from __future__ import annotations

from pathlib import Path

from app import propose, registry, workspace
from helpers import write_rows


def _meta(tmp_path: Path, name: str, header: list[str], rows: list[list]):
    path = tmp_path / name
    write_rows(path, header, rows)
    return workspace.inspect_table(path)


def _dataset_rows(n=12, extra_columns=None):
    rows = []
    for index in range(n):
        row = [f"d{index}", f"observation {index} with enough words to be text"]
        row.extend(extra_columns or [])
        rows.append(row)
    header = ["doc_id", "comment"] + (["source"] if extra_columns else [])
    return header, rows


def test_clean_schema_is_proposed_without_blocking_problems(tmp_path):
    header, rows = _dataset_rows()
    dataset = _meta(tmp_path, "data.csv", header, rows)
    benchmark = _meta(tmp_path, "labels.csv", ["ref_id", "sentiment"],
                      [[f"d{i}", "positive" if i % 2 else "negative"] for i in range(12)])
    proposal = propose.build_proposal(dataset, benchmark)
    assert proposal.dataset_id_column == "doc_id"
    assert proposal.text_column == "comment"
    assert proposal.benchmark_id_column == "ref_id"
    assert proposal.label_column == "sentiment"
    assert [c["name"] for c in proposal.classes] == ["negative", "positive"]
    assert proposal.label_map == {"negative": "negative", "positive": "positive"}
    assert proposal.blocking() == []
    # Definitions are placeholders, so this is a decision, not a blocker.
    assert any("definition" in p.message for p in proposal.decisions())
    assert proposal.max_calls > 0
    assert 0 < proposal.max_usd <= 10


def test_numeric_labels_get_generic_categories_and_never_infer_polarity(tmp_path):
    header, rows = _dataset_rows()
    dataset = _meta(tmp_path, "data.csv", header, rows)
    benchmark = _meta(tmp_path, "labels.csv", ["ref_id", "sentiment"],
                      [[f"d{i}", "1" if i % 2 else "0"] for i in range(12)])
    proposal = propose.build_proposal(dataset, benchmark)
    assert proposal.blocking() == []
    assert all(c["name"].startswith("category-") for c in proposal.classes)
    assert proposal.label_map == {"0": "category-1", "1": "category-2"}
    assert any("polarity is never guessed" in p.fix_hint
               for p in proposal.decisions())


def test_ambiguous_text_columns_are_flagged(tmp_path):
    rows = [[f"d{i}",
             f"first text column {i} with plenty of words to look like free text",
             f"second text column {i} also with plenty of words to look like text"]
            for i in range(12)]
    dataset = _meta(tmp_path, "data.csv", ["doc_id", "alpha", "beta"], rows)
    benchmark = _meta(tmp_path, "labels.csv", ["ref_id", "sentiment"],
                      [[f"d{i}", "positive" if i % 2 else "negative"] for i in range(12)])
    proposal = propose.build_proposal(dataset, benchmark)
    assert proposal.text_column is None
    assert any(p.field == "text_column" and p.severity == "blocking"
               for p in proposal.problems)


def test_mismatched_ids_are_a_decision_with_counts(tmp_path):
    header, rows = _dataset_rows(12)
    dataset = _meta(tmp_path, "data.csv", header, rows)
    benchmark = _meta(tmp_path, "labels.csv", ["ref_id", "sentiment"],
                      [[f"d{i}", "positive" if i % 2 else "negative"]
                       for i in range(0, 10)])  # d10, d11 missing
    proposal = propose.build_proposal(dataset, benchmark)
    assert any(p.field == "benchmark.on_id_mismatch" and p.severity == "decision"
               for p in proposal.problems)
    assert [option["action"] for option in
            next(p for p in proposal.problems
                 if p.field == "benchmark.on_id_mismatch").options] == \
        ["error", "inner_join"]


def test_duplicate_ids_and_blank_text_are_blocking(tmp_path):
    rows = [[f"d{i}", f"observation {i} with enough words to be recognised as text"]
            for i in range(59)]
    rows.append(["d0", f"observation 59 with enough words to be text"])  # duplicate ID
    rows[3] = ["d3", "   "]                                              # blank text
    dataset = _meta(tmp_path, "data.csv", ["doc_id", "comment"], rows)
    benchmark = _meta(tmp_path, "labels.csv", ["ref_id", "sentiment"],
                      [[row[0], "positive" if i % 2 else "negative"]
                       for i, row in enumerate(rows)])
    proposal = propose.build_proposal(dataset, benchmark)
    messages = " ".join(p.message for p in proposal.blocking())
    assert "Duplicate IDs" in messages
    assert "empty text" in messages


def test_budget_respects_the_registry_ceiling(tmp_path):
    header, rows = _dataset_rows(50)
    dataset = _meta(tmp_path, "data.csv", header, rows)
    benchmark = _meta(tmp_path, "labels.csv", ["ref_id", "sentiment"],
                      [[f"d{i}", "positive" if i % 2 else "negative"] for i in range(50)])
    proposal = propose.build_proposal(dataset, benchmark)
    provider = registry.enabled_providers()[0]
    ceiling = registry.max_calls_ceiling(provider["models"][0])
    assert proposal.max_calls <= ceiling

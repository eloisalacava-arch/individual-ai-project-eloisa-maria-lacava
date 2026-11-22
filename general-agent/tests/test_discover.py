"""Column discovery and the clarification policy.

The tests assert the policy rather than one implementation: unambiguous inputs
resolve silently, and each genuinely ambiguous situation produces a question
that names the config field which resolves it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from general_agent.config import load_config
from general_agent.discover import map_labels, propose
from general_agent.io import read_table
from helpers import make_config


def _load(tmp_path, **kwargs):
    cfg = load_config(make_config(tmp_path, **kwargs))
    dataset = read_table(cfg["dataset"]["path"])
    benchmark = read_table(cfg["benchmark"]["path"])
    return cfg, dataset, benchmark


def test_unambiguous_inputs_resolve_without_questions(tmp_path):
    cfg, dataset, benchmark = _load(tmp_path)
    proposal = propose(dataset, benchmark, cfg)
    assert proposal.ready
    assert proposal.mapping["id_column"] == "review_id"
    assert proposal.mapping["text_column"] == "review_text"
    assert proposal.mapping["label_column"] == "sentiment"
    assert proposal.decisions


def test_ambiguous_text_columns_are_asked_about(tmp_path):
    cfg, dataset, benchmark = _load(
        tmp_path,
        dataset="dataset_messy.csv",
        benchmark="benchmark_messy.csv",
        patch={
            "dataset": {"id_column": None, "text_column": None},
            "benchmark": {"id_column": None, "label_column": "label"},
        },
    )
    proposal = propose(dataset, benchmark, cfg)
    assert not proposal.ready
    fields = {question["field"] for question in proposal.questions}
    assert fields == {"dataset.text_column"}
    options = proposal.questions[0]["options"]
    assert {option["column"] for option in options} == {"headline", "body"}


def test_id_column_is_resolved_from_a_name_hint_among_candidates(tmp_path):
    cfg, dataset, benchmark = _load(
        tmp_path,
        dataset="dataset_messy.csv",
        benchmark="benchmark_messy.csv",
        patch={
            "dataset": {"id_column": None, "text_column": None},
            "benchmark": {"id_column": None, "label_column": "label"},
        },
    )
    proposal = propose(dataset, benchmark, cfg)
    assert proposal.mapping["id_column"] == "article_id"


def test_numeric_labels_without_a_map_are_a_question(tmp_path):
    cfg, dataset, benchmark = _load(
        tmp_path, benchmark="benchmark_numeric.csv",
        patch={"benchmark": {"label_column": None}},
    )
    proposal = propose(dataset, benchmark, cfg)
    assert not proposal.ready
    assert any(q["field"] == "benchmark.label_map" for q in proposal.questions)


def test_numeric_labels_with_a_map_resolve(tmp_path):
    cfg, dataset, benchmark = _load(
        tmp_path, benchmark="benchmark_numeric.csv",
        patch={"benchmark": {
            "label_column": "label_code",
            "label_map": {"0": "negative", "1": "positive"},
        }},
    )
    proposal = propose(dataset, benchmark, cfg)
    assert proposal.ready
    assert any("label_map applied" in decision for decision in proposal.decisions)


def test_id_mismatch_is_a_question_by_default(tmp_path):
    cfg, dataset, benchmark = _load(tmp_path, benchmark="benchmark_subset.csv")
    proposal = propose(dataset, benchmark, cfg)
    assert not proposal.ready
    assert any(q["field"] == "benchmark.on_id_mismatch" for q in proposal.questions)
    assert proposal.stats["id_alignment"]["n_only_dataset"] == 2


def test_id_mismatch_resolves_when_inner_join_is_configured(tmp_path):
    cfg, dataset, benchmark = _load(
        tmp_path, benchmark="benchmark_subset.csv",
        patch={"benchmark": {"on_id_mismatch": "inner_join"}},
    )
    proposal = propose(dataset, benchmark, cfg)
    assert proposal.ready
    assert any("inner join approved by config" in decision
               for decision in proposal.decisions)


def test_duplicate_ids_are_a_question(tmp_path):
    cfg, dataset, benchmark = _load(tmp_path, dataset="dataset_duplicate.csv")
    proposal = propose(dataset, benchmark, cfg)
    assert not proposal.ready
    assert any(q["field"] == "benchmark.allow_duplicate_ids" for q in proposal.questions)
    assert proposal.stats["duplicate_ids"]["dataset"] == 1


def test_duplicates_allowed_by_config_become_a_recorded_decision(tmp_path):
    cfg, dataset, benchmark = _load(
        tmp_path, dataset="dataset_duplicate.csv",
        patch={"benchmark": {"allow_duplicate_ids": True, "on_id_mismatch": "inner_join"}},
    )
    proposal = propose(dataset, benchmark, cfg)
    assert proposal.ready
    assert any("duplicate IDs present" in decision for decision in proposal.decisions)


def test_map_labels_identity_and_unknown_values():
    classes = ["positive", "negative"]
    identity, unmapped = map_labels(
        pd.Series(["positive", "negative", "positive"]), None, classes
    )
    assert unmapped == []
    assert identity.tolist() == ["positive", "negative", "positive"]

    mapped, unmapped = map_labels(
        pd.Series(["1", "0", "9"]), {"0": "negative", "1": "positive"}, classes
    )
    assert unmapped == ["9"]
    assert mapped.tolist()[:2] == ["positive", "negative"]
    assert pd.isna(mapped.iloc[2])


def test_label_map_passes_through_values_that_are_already_canonical():
    classes = ["positive", "negative"]
    mapped, unmapped = map_labels(
        pd.Series(["positive", "1", "0"]),
        {"0": "negative", "1": "positive"},
        classes,
    )
    assert unmapped == []
    assert mapped.tolist() == ["positive", "positive", "negative"]


def test_unknown_label_becomes_a_recorded_exclusion_when_policy_is_exclude(tmp_path):
    cfg, dataset, benchmark = _load(
        tmp_path, benchmark="benchmark_numeric.csv",
        patch={"benchmark": {
            "label_column": "label_code",
            "on_unknown_label": "exclude",
        }},
    )
    proposal = propose(dataset, benchmark, cfg)
    assert proposal.ready
    assert any("excluded with counts" in decision for decision in proposal.decisions)
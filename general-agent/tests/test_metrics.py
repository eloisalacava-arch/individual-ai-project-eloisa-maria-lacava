"""Metric definitions, checked against hand-computed examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from general_agent.evaluate import metrics


def test_binary_kappa_matches_a_hand_computed_table():
    # [[20, 5], [10, 15]]: po = 0.7, row = (0.5, 0.5), col = (0.6, 0.4)
    # pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = (0.7 - 0.5) / 0.5 = 0.4
    matrix = np.array([[20, 5], [10, 15]])
    assert metrics.accuracy(matrix) == pytest.approx(0.7)
    assert metrics.cohen_kappa(matrix) == pytest.approx(0.4)


def test_perfect_agreement_gives_accuracy_and_kappa_one():
    matrix = np.array([[10, 0, 0], [0, 8, 0], [0, 0, 6]])
    assert metrics.accuracy(matrix) == pytest.approx(1.0)
    assert metrics.cohen_kappa(matrix) == pytest.approx(1.0)
    assert metrics.weighted_kappa(matrix, "quadratic") == pytest.approx(1.0)


def test_complete_disagreement_in_a_balanced_binary_table_is_minus_one():
    matrix = np.array([[0, 10], [10, 0]])
    assert metrics.accuracy(matrix) == pytest.approx(0.0)
    assert metrics.cohen_kappa(matrix) == pytest.approx(-1.0)
    assert metrics.weighted_kappa(matrix, "quadratic") == pytest.approx(-1.0)


def test_weighted_kappa_rewards_near_misses():
    # All errors are adjacent steps: quadratic weighting must be kinder than
    # unweighted kappa.
    matrix = np.array([[10, 1, 0], [1, 10, 1], [0, 1, 10]])
    unweighted = metrics.cohen_kappa(matrix)
    weighted = metrics.weighted_kappa(matrix, "quadratic")
    assert weighted > unweighted
    assert 0 < unweighted < 1


def test_weighted_kappa_is_not_inverted_for_opposite_errors():
    # Errors are all two steps apart; weighted kappa must be negative, not +1.
    matrix = np.array([[0, 0, 10], [0, 0, 0], [10, 0, 0]])
    assert metrics.weighted_kappa(matrix, "quadratic") < 0


def test_per_class_counts_and_scores():
    matrix = np.array([[8, 1, 1], [2, 7, 1], [1, 2, 7]])
    rows = metrics.per_class(matrix)
    assert [entry["support"] for entry in rows] == [10, 10, 10]
    assert [entry["predicted"] for entry in rows] == [11, 10, 9]
    assert rows[0]["tp"] == 8 and rows[0]["fp"] == 3 and rows[0]["fn"] == 2
    assert rows[0]["precision"] == pytest.approx(8 / 11)
    assert rows[0]["recall"] == pytest.approx(0.8)
    assert rows[0]["f1"] == pytest.approx(
        2 * (8 / 11) * 0.8 / ((8 / 11) + 0.8)
    )


def test_f1_is_zero_when_precision_is_zero_not_nan():
    # Class 2 has support but every prediction of it is wrong.
    matrix = np.array([[3, 0, 2], [0, 5, 0], [0, 5, 0]])
    rows = metrics.per_class(matrix)
    assert rows[2]["support"] == 5
    assert rows[2]["predicted"] == 2
    assert rows[2]["precision"] == pytest.approx(0.0)
    assert rows[2]["recall"] == pytest.approx(0.0)
    assert rows[2]["f1"] == pytest.approx(0.0)


def test_balanced_accuracy_averages_recall_across_classes():
    matrix = np.array([[10, 0, 0], [0, 5, 5], [0, 0, 0]])
    # recalls: 1.0, 0.5, 0.0 (class 2 has zero support and is skipped)
    assert metrics.balanced_accuracy(matrix) == pytest.approx(0.75)


def test_ordinal_summary_counts_distances_and_buckets():
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    summary = metrics.ordinal_summary(y_true, y_pred, 3)
    assert summary["buckets"] == {"agree": 3, "adjacent": 1, "opposite": 1}
    assert summary["distances"] == {0: 3, 1: 1, 2: 1}


def test_label_indices_rejects_unknown_labels():
    with pytest.raises(ValueError):
        metrics.label_indices(["positive", "mixed"], ["positive", "negative"])


def test_empty_matrix_is_not_a_number():
    assert math.isnan(metrics.accuracy(np.zeros((3, 3), dtype=int)))
    assert math.isnan(metrics.cohen_kappa(np.zeros((3, 3), dtype=int)))

def test_binary_severity_does_not_double_count_errors():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    summary = metrics.ordinal_summary(y_true, y_pred, 2)
    assert summary["buckets"]["agree"] == 2
    assert summary["buckets"]["adjacent"] == 0
    assert summary["buckets"]["opposite"] == 2

"""Interval estimates: closed-form Wilson and seeded bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from general_agent.evaluate import intervals, metrics


def test_wilson_matches_a_known_example():
    low, high = intervals.wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_is_bounded_and_handles_extremes():
    assert intervals.wilson(0, 10)[0] == pytest.approx(0.0)
    assert intervals.wilson(10, 10)[1] == pytest.approx(1.0)
    low, high = intervals.wilson(0, 10)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_empty_sample_is_not_a_number():
    low, high = intervals.wilson(0, 0)
    assert np.isnan(low) and np.isnan(high)


def _sample():
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 2, 1, 2, 1])
    return y_true, y_pred


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    y_true, y_pred = _sample()
    first = intervals.bootstrap_core(y_true, y_pred, 3, draws=200, seed=42)
    second = intervals.bootstrap_core(y_true, y_pred, 3, draws=200, seed=42)
    assert first == second
    third = intervals.bootstrap_core(y_true, y_pred, 3, draws=200, seed=43)
    assert third["kappa_unweighted"]["low"] != first["kappa_unweighted"]["low"]


def test_bootstrap_reports_valid_draw_counts():
    y_true, y_pred = _sample()
    result = intervals.bootstrap_core(y_true, y_pred, 3, draws=100, seed=7)
    for key in ("accuracy", "kappa_unweighted", "kappa_quadratic"):
        assert 0 < result[key]["valid"] <= 100
        assert result[key]["low"] <= result[key]["high"]


def test_bootstrap_interval_contains_the_point_estimate():
    y_true, y_pred = _sample()
    matrix = metrics.confusion_matrix(y_true, y_pred, 3)
    point = metrics.cohen_kappa(matrix)
    result = intervals.bootstrap_core(y_true, y_pred, 3, draws=500, seed=20261116)
    assert result["kappa_unweighted"]["low"] <= point <= result["kappa_unweighted"]["high"]


def test_per_class_bootstrap_shape():
    y_true, y_pred = _sample()
    result = intervals.bootstrap_per_class(y_true, y_pred, 3, draws=100, seed=1)
    assert len(result) == 3
    for entry in result:
        for key in ("precision", "recall", "f1"):
            assert entry[f"{key}_low"] <= entry[f"{key}_high"]
            assert entry[f"{key}_valid"] <= 100
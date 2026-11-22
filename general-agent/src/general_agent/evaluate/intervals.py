"""Confidence intervals.

Two methods, chosen by whether the quantity is a binomial proportion:

  * Wilson score interval  - overall accuracy, per-class recall, ordinal
    distance shares. Closed form, validated in the parent project.
  * Seeded percentile bootstrap - kappa (unweighted and weighted), balanced
    accuracy, per-class precision and F1, where no simple binomial model
    applies. Sampling with replacement is the same device the parent project
    used to catch its own inflated kappa standard error (M8).

Point estimates always come from the full sample. Bootstrap draws that are
undefined (for example a kappa whose expected agreement is 1 in a resample) are
dropped and counted, never silently imputed.
"""

from __future__ import annotations

import math

import numpy as np
from statistics import NormalDist

from . import metrics


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_core(y_true: np.ndarray, y_pred: np.ndarray, k: int,
                   draws: int, seed: int, level: float = 0.95) -> dict:
    """Percentile CIs for the matrix-derived metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: dict[str, list[float]] = {
        "accuracy": [], "balanced_accuracy": [],
        "kappa_unweighted": [], "kappa_quadratic": [],
    }
    for _ in range(draws):
        indices = rng.integers(0, n, n)
        matrix = metrics.confusion_matrix(y_true[indices], y_pred[indices], k)
        values = metrics.all_core_metrics(matrix)
        for key, value in values.items():
            if value is not None and not math.isnan(value):
                samples[key].append(float(value))

    tail = (1 - level) / 2
    result = {}
    for key, values in samples.items():
        if not values:
            result[key] = {"low": math.nan, "high": math.nan, "valid": 0}
            continue
        array = np.array(values)
        result[key] = {
            "low": float(np.quantile(array, tail)),
            "high": float(np.quantile(array, 1 - tail)),
            "valid": len(values),
        }
    return result


def bootstrap_per_class(y_true: np.ndarray, y_pred: np.ndarray, k: int,
                        draws: int, seed: int, level: float = 0.95) -> list[dict]:
    """Percentile CIs for per-class precision, recall and F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collected = [{"precision": [], "recall": [], "f1": []} for _ in range(k)]
    for _ in range(draws):
        indices = rng.integers(0, n, n)
        matrix = metrics.confusion_matrix(y_true[indices], y_pred[indices], k)
        for index, entry in enumerate(metrics.per_class(matrix)):
            for key in ("precision", "recall", "f1"):
                value = entry[key]
                if value is not None and not math.isnan(value):
                    collected[index][key].append(float(value))

    tail = (1 - level) / 2
    result = []
    for index in range(k):
        entry = {}
        for key in ("precision", "recall", "f1"):
            values = collected[index][key]
            if not values:
                entry[f"{key}_low"] = entry[f"{key}_high"] = math.nan
                entry[f"{key}_valid"] = 0
            else:
                array = np.array(values)
                entry[f"{key}_low"] = float(np.quantile(array, tail))
                entry[f"{key}_high"] = float(np.quantile(array, 1 - tail))
                entry[f"{key}_valid"] = len(values)
        result.append(entry)
    return result
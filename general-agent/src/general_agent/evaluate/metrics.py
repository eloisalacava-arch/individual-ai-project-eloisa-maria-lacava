"""Evaluation metrics for single-label classification against a reference.

Everything is computed from one confusion matrix with a fixed orientation:

    rows = human reference, columns = AI prediction

That orientation is stated in every table header so a reader can never mistake
a cell for its transpose (the parent project's M6/M7 failure).

Kappa variants follow Cohen (1960) and the standard weighted formulation:
``kappa_w = 1 - D_o / D_e`` with quadratic weights ``1 - (i-j)^2/(k-1)^2``.
"""

from __future__ import annotations

import math

import numpy as np


def label_indices(values, classes: list[str]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(classes)}
    unknown = sorted({str(value) for value in values} - set(mapping))
    if unknown:
        raise ValueError(f"labels outside the declared classes: {unknown}")
    return np.array([mapping[str(value)] for value in values], dtype=int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=int)
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[int(true_index), int(pred_index)] += 1
    return matrix


def accuracy(matrix: np.ndarray) -> float:
    n = matrix.sum()
    return float(np.trace(matrix) / n) if n else math.nan


def cohen_kappa(matrix: np.ndarray) -> float:
    n = matrix.sum()
    if n == 0:
        return math.nan
    observed = np.trace(matrix) / n
    row = matrix.sum(axis=1) / n
    col = matrix.sum(axis=0) / n
    expected = float(np.dot(row, col))
    if expected >= 1.0:
        return math.nan
    return float((observed - expected) / (1.0 - expected))


def weighted_kappa(matrix: np.ndarray, kind: str = "quadratic") -> float:
    """Weighted kappa for ordinal classes. Weights 1 for perfect, 0 for opposite."""
    k = matrix.shape[0]
    if k < 2:
        return math.nan
    n = matrix.sum()
    if n == 0:
        return math.nan
    indices = np.arange(k)
    gaps = np.abs(indices[:, None] - indices[None, :]).astype(float)
    if kind == "quadratic":
        weights = 1.0 - (gaps / (k - 1)) ** 2
    elif kind == "linear":
        weights = 1.0 - (gaps / (k - 1))
    else:
        raise ValueError(f"unknown weighting {kind!r}")

    # weights are AGREEMENT weights: 1 on the diagonal, 0 at maximum distance.
    # kappa_w = (P_o^w - P_e^w) / (1 - P_e^w). Writing this as 1 - D_o/D_e with
    # the same weights would invert the sign for maximally disagreeing data.
    row = matrix.sum(axis=1) / n
    col = matrix.sum(axis=0) / n
    observed_weighted = float((weights * matrix).sum() / n)
    expected_weighted = float((weights * np.outer(row, col)).sum())
    if expected_weighted >= 1.0:
        return math.nan
    return (observed_weighted - expected_weighted) / (1.0 - expected_weighted)


def per_class(matrix: np.ndarray) -> list[dict]:
    n = matrix.sum()
    row = matrix.sum(axis=1)
    col = matrix.sum(axis=0)
    results = []
    for index in range(matrix.shape[0]):
        tp = int(matrix[index, index])
        support = int(row[index])
        predicted = int(col[index])
        fp = predicted - tp
        fn = support - tp
        precision = tp / predicted if predicted else math.nan
        recall = tp / support if support else math.nan
        if not math.isnan(precision) and not math.isnan(recall):
            denominator = precision + recall
            f1 = 2 * precision * recall / denominator if denominator else 0.0
        else:
            f1 = math.nan
        results.append({
            "tp": tp,
            "fp": int(fp),
            "fn": int(fn),
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return results


def balanced_accuracy(matrix: np.ndarray) -> float:
    recalls = [entry["recall"] for entry in per_class(matrix)]
    valid = [value for value in recalls if not math.isnan(value)]
    return float(np.mean(valid)) if valid else math.nan


def ordinal_summary(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    distances = np.abs(y_true - y_pred)
    total = len(distances)
    # For a binary task the only possible disagreement is distance 1, which is
    # simultaneously the maximum distance. Counting it as "adjacent" as well
    # would double-count every error, so adjacency is only defined for k >= 3.
    buckets = {
        "agree": int((distances == 0).sum()),
        "adjacent": int((distances == 1).sum()) if k > 2 else 0,
        "opposite": int((distances == k - 1).sum()) if k > 1 else 0,
    }
    return {
        "n": total,
        "distances": {int(d): int((distances == d).sum()) for d in range(k)},
        "buckets": buckets,
    }


def all_core_metrics(matrix: np.ndarray) -> dict:
    """One dict used both for point estimates and inside the bootstrap."""
    return {
        "accuracy": accuracy(matrix),
        "balanced_accuracy": balanced_accuracy(matrix),
        "kappa_unweighted": cohen_kappa(matrix),
        "kappa_quadratic": weighted_kappa(matrix, "quadratic")
        if matrix.shape[0] > 1 else math.nan,
    }
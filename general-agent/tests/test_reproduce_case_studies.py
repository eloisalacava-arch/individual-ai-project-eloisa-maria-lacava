"""Independent recomputation of the shipped case-study statistics.

Accuracy and Cohen's kappa are recomputed from the frozen predictions and the
sealed benchmark with a local implementation, then compared with the shipped
result tables (``t02``, ``t03``, ``t13``). The engine's ``evaluate.metrics``
functions are deliberately not imported, so a defect there cannot hide here.

The IMDb targets come from the parent project's verified outputs (SPEC §5):
AI-vs-manual kappa 0.9399, AI-vs-IMDb kappa 0.8899, manual-vs-IMDb kappa 0.850.

These tests skip when the case-study runs are not present in the checkout.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

RUNS = Path(__file__).resolve().parents[1] / "runs"


def _load(run_id: str):
    run = RUNS / run_id
    frozen_path = run / "results" / "frozen_predictions.csv"
    sealed_path = run / "sealed" / "benchmark.csv"
    if not frozen_path.exists() or not sealed_path.exists():
        pytest.skip(f"{run_id} case study is not present in this checkout")
    return run, pd.read_csv(frozen_path, dtype="string"), pd.read_csv(sealed_path, dtype="string")


def _accuracy(prediction: pd.Series, reference: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame({"prediction": prediction, "reference": reference}).dropna()
    return float((frame["prediction"] == frame["reference"]).mean()), int(len(frame))


def _cohen_kappa(prediction: pd.Series, reference: pd.Series) -> float:
    frame = pd.DataFrame({"prediction": prediction, "reference": reference}).dropna()
    n = len(frame)
    if n == 0:
        return math.nan
    labels = sorted(set(frame["prediction"]) | set(frame["reference"]))
    prediction_counts = {label: 0 for label in labels}
    reference_counts = {label: 0 for label in labels}
    agree = 0
    for predicted, actual in zip(frame["prediction"], frame["reference"]):
        prediction_counts[predicted] += 1
        reference_counts[actual] += 1
        if predicted == actual:
            agree += 1
    observed = agree / n
    expected = sum((prediction_counts[c] / n) * (reference_counts[c] / n)
                   for c in labels)
    return (observed - expected) / (1 - expected)


def _shipped_kappa(run: Path, metric_contains: str = "unweighted") -> float:
    table = pd.read_csv(run / "results" / "t03_kappa.csv")
    return float(table.loc[table["metric"].str.contains(metric_contains), "value"].iloc[0])


def test_imdb_accuracy_and_kappa_reproduce():
    run, frozen, sealed = _load("imdb_case_study")
    frame = frozen.rename(columns={"label": "prediction"}).merge(
        sealed[["id", "human_sentiment"]], on="id", how="inner")
    frame = frame[frame["status"] == "ok"]
    accuracy, n = _accuracy(frame["prediction"], frame["human_sentiment"])
    kappa = _cohen_kappa(frame["prediction"], frame["human_sentiment"])

    shipped = pd.read_csv(run / "results" / "t02_accuracy.csv").iloc[0]
    assert n == int(shipped["n"]) == 200
    assert accuracy == pytest.approx(float(shipped["accuracy"]), abs=1e-6)
    assert kappa == pytest.approx(_shipped_kappa(run), abs=1e-5)


def test_imdb_reference_agreement_reproduces_the_verified_targets():
    run, frozen, sealed = _load("imdb_case_study")
    frame = frozen.rename(columns={"label": "prediction"}).merge(
        sealed[["id", "human_sentiment", "imdb_label"]], on="id", how="inner")
    frame = frame[frame["status"] == "ok"]
    proxy = frame["imdb_label"].map({"0": "negative", "1": "positive"})

    ai_manual = _cohen_kappa(frame["prediction"], frame["human_sentiment"])
    ai_proxy = _cohen_kappa(frame["prediction"], proxy)
    manual_proxy = _cohen_kappa(frame["human_sentiment"], proxy)

    # Target 1: AI vs the manual annotation (the primary reference).
    assert ai_manual == pytest.approx(_shipped_kappa(run), abs=1e-5)
    assert ai_manual == pytest.approx(0.9399, abs=1e-3)
    # Targets 2 and 3: the parent project's verified AI-vs-IMDb and
    # manual-vs-IMDb kappas.
    shipped = pd.read_csv(run / "results" / "t13_reference_agreement.csv").set_index(
        "comparison")
    assert ai_proxy == pytest.approx(
        float(shipped.loc["ai_vs_imdb_label", "kappa_unweighted"]), abs=1e-5)
    assert ai_proxy == pytest.approx(0.8899, abs=1e-3)
    assert manual_proxy == pytest.approx(
        float(shipped.loc["human_sentiment_vs_imdb_label", "kappa_unweighted"]), abs=1e-5)
    assert manual_proxy == pytest.approx(0.850, abs=1e-3)


def test_phrasebank_accuracy_and_kappa_reproduce():
    run, frozen, sealed = _load("phrasebank_75agree")
    frame = frozen.rename(columns={"label": "prediction"}).merge(
        sealed[["id", "label"]].rename(columns={"label": "reference"}),
        on="id", how="inner")
    frame = frame[frame["status"] == "ok"]
    accuracy, n = _accuracy(frame["prediction"], frame["reference"])
    kappa = _cohen_kappa(frame["prediction"], frame["reference"])

    shipped = pd.read_csv(run / "results" / "t02_accuracy.csv").iloc[0]
    assert n == int(shipped["n"]) == 300
    assert accuracy == pytest.approx(float(shipped["accuracy"]), abs=1e-6)
    assert kappa == pytest.approx(_shipped_kappa(run), abs=1e-5)

    # No secondary reference is declared for this study, so the reference-agreement
    # table is empty (header only).
    references = pd.read_csv(run / "results" / "t13_reference_agreement.csv")
    assert len(references) == 0

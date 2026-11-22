"""Figures for the PDF reports, regenerated from the verified tables.

Presentation only. Every series comes from a CSV the engine already wrote; no
statistic is recomputed, and the SVG figures produced by stage 07 are left
untouched.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ACCENT = "#2F5D8A"
MUTED = "#6B7280"


def _png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _result(cfg: dict, name: str) -> pd.DataFrame:
    path = Path(cfg["_meta"]["output_dir"]) / "results" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run stages 05-06 first")
    return pd.read_csv(path)


def confusion_matrix_png(cfg: dict) -> bytes:
    frame = pd.read_csv(Path(cfg["_meta"]["output_dir"]) / "results"
                        / "t04_confusion_matrix.csv", index_col=0)
    frame = frame.drop(index="total_predicted", errors="ignore")
    frame = frame.drop(columns="total_human", errors="ignore")
    values = frame.to_numpy(float)
    figure, axis = plt.subplots(figsize=(4.6, 3.8))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(frame.columns)), frame.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(frame.index)), frame.index)
    axis.set_xlabel("AI answer")
    axis.set_ylabel("Human label")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            count = int(values[row, column])
            if count:
                axis.text(column, row, str(count), ha="center", va="center",
                          color="white" if values[row, column] > values.max() / 2
                          else "#1F2933", fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return _png(figure)


def per_class_recall_png(cfg: dict) -> bytes:
    frame = _result(cfg, "t05_per_class_metrics.csv")
    values = frame["recall"].to_numpy(float)
    lower = values - frame["recall_ci_low"].to_numpy(float)
    upper = frame["recall_ci_high"].to_numpy(float) - values
    figure, axis = plt.subplots(figsize=(5.2, 3.2))
    axis.bar(frame["class"], values, yerr=[lower, upper], capsize=3,
             color=ACCENT, alpha=0.85)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Recall (95% CI)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return _png(figure)


# --------------------------------------------------------------------------- #
# Comparison figures (from comparison/tables/)                                #
# --------------------------------------------------------------------------- #

def _comparison_table(directory: Path, name: str) -> pd.DataFrame:
    path = Path(directory) / "tables" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run stage 07 first")
    return pd.read_csv(path)


def comparison_metrics_png(directory: Path) -> bytes:
    metrics = _comparison_table(directory, "cross_dataset_metrics.csv")
    labels = metrics["dataset"].tolist()
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for column, low, high, title, axis in (
        ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "Accuracy", axes[0]),
        ("kappa_unweighted", "kappa_ci_low", "kappa_ci_high", "Cohen's kappa",
         axes[1]),
    ):
        values = metrics[column].to_numpy(float)
        axis.errorbar(values, range(len(labels)),
                      xerr=[values - metrics[low].to_numpy(float),
                            metrics[high].to_numpy(float) - values],
                      fmt="o", capsize=4, color=ACCENT)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlim(0, 1.0)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    return _png(figure)


def comparison_class_recall_png(directory: Path) -> bytes:
    frame = _comparison_table(directory, "cross_dataset_classes.csv")
    datasets = frame["dataset"].unique().tolist()
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    width = 0.8 / max(len(datasets), 1)
    for offset, dataset in enumerate(datasets):
        subset = frame[frame["dataset"] == dataset]
        positions = [index + offset * width for index in range(len(subset))]
        values = subset["recall"].to_numpy(float)
        axis.bar(positions, values, width=width, label=dataset,
                 yerr=[values - subset["recall_ci_low"].to_numpy(float),
                       subset["recall_ci_high"].to_numpy(float) - values],
                 capsize=3, alpha=0.85)
        axis.set_xticks([index + width * (len(datasets) - 1) / 2
                         for index in range(len(subset))],
                        subset["class"], rotation=20, ha="right")
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Recall (95% CI)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return _png(figure)


def comparison_severity_png(directory: Path) -> bytes:
    frame = _comparison_table(directory, "cross_dataset_severity.csv")
    buckets = ["agree", "adjacent", "opposite"]
    colours = {"agree": "#4c9f70", "adjacent": "#d9a441", "opposite": "#b5446e"}
    datasets = frame["dataset"].unique().tolist()
    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    bottom = [0.0] * len(datasets)
    for bucket in buckets:
        shares = []
        for dataset in datasets:
            subset = frame[(frame["dataset"] == dataset) & (frame["bucket"] == bucket)]
            shares.append(float(subset["share"].iloc[0]) if len(subset) else 0.0)
        axis.bar(datasets, shares, bottom=bottom, label=bucket, color=colours[bucket])
        bottom = [current + share for current, share in zip(bottom, shares)]
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Share of observations")
    # The stacked bars fill the axes, so place the legend outside it: a legend
    # anchored inside would overlap the bars.
    axis.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False)
    figure.tight_layout()
    return _png(figure)

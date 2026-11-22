"""Deterministic aggregation of explanations into pattern tables.

No model call happens here. Every count and share is computed from
``explanations.csv``; the control sample is reported separately and contrasted,
never pooled with the disagreements.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from .settings import QualitativeError

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def read_explanations(out_dir: Path) -> pd.DataFrame:
    path = Path(out_dir) / "explanations.csv"
    if not path.exists():
        raise QualitativeError(f"{path} is missing; run the explanations first")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    frame["reason_list"] = frame["reasons"].apply(
        lambda value: [part for part in str(value).split(";") if part])
    return frame


def _explained(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (frame["status"] == "ok") & (frame["summary"].str.strip() != "")
    return frame[mask].copy()


def _selected_counts(out_dir: Path) -> dict:
    path = Path(out_dir) / "selection_meta.json"
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
        return {
            "disagreement": int(meta.get("n_disagreements", 0)),
            "agreement": int(meta.get("n_agreements", 0)),
        }
    selected = pd.read_csv(Path(out_dir) / "selected.csv", dtype="string")
    counts = selected["group"].value_counts().to_dict()
    return {"disagreement": int(counts.get("disagreement", 0)),
            "agreement": int(counts.get("agreement", 0))}


def _coverage(frame: pd.DataFrame, selected: dict) -> pd.DataFrame:
    rows = []
    for group in ("disagreement", "agreement"):
        subset = frame[frame["group"] == group]
        explained = _explained(subset)
        statuses = subset["status"].value_counts().to_dict()
        rows.append({
            "group": group,
            "n_selected": int(selected.get(group, 0)),
            "n_explained": int(len(explained)),
            "n_failed": int(len(subset) - len(explained)),
            "explained_share": (len(explained) / len(subset)) if len(subset) else 0.0,
            "statuses": "; ".join(f"{k}={v}" for k, v in sorted(statuses.items())),
        })
    return pd.DataFrame(rows)


def _patterns(frame: pd.DataFrame, selected: dict) -> pd.DataFrame:
    rows = []
    for group in ("disagreement", "agreement"):
        subset = _explained(frame[frame["group"] == group])
        total = len(subset)
        selected_n = int(selected.get(group, 0))
        counts: dict[str, int] = {}
        directions: dict[str, dict[str, int]] = {}
        for _, row in subset.iterrows():
            for reason in row["reason_list"]:
                counts[reason] = counts.get(reason, 0) + 1
                directions.setdefault(reason, {})
                key = str(row.get("direction", ""))
                directions[reason][key] = directions[reason].get(key, 0) + 1
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            top = sorted(directions.get(reason, {}).items(),
                         key=lambda item: (-item[1], item[0]))[:3]
            rows.append({
                "group": group,
                "reason": reason,
                "n": count,
                "share_of_explained": count / total if total else 0.0,
                "share_of_selected": count / selected_n if selected_n else 0.0,
                "top_directions": "; ".join(f"{key} ({value})" for key, value in top),
            })
    return pd.DataFrame(rows, columns=[
        "group", "reason", "n", "share_of_explained", "share_of_selected",
        "top_directions"])


def _cooccurrence(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ("disagreement", "agreement"):
        subset = _explained(frame[frame["group"] == group])
        total = len(subset)
        counts: dict[tuple[str, str], int] = {}
        for _, row in subset.iterrows():
            for pair in combinations(sorted(set(row["reason_list"])), 2):
                counts[pair] = counts.get(pair, 0) + 1
        for (left, right), count in sorted(counts.items(),
                                           key=lambda item: (-item[1], item[0])):
            rows.append({
                "group": group,
                "reason_a": left,
                "reason_b": right,
                "n": count,
                "share_of_explained": count / total if total else 0.0,
            })
    return pd.DataFrame(rows, columns=["group", "reason_a", "reason_b", "n",
                                       "share_of_explained"])


def _reason_shares(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    shares: dict[str, dict[str, float]] = {}
    for group in ("disagreement", "agreement"):
        subset = _explained(frame[frame["group"] == group])
        total = len(subset)
        counts: dict[str, int] = {}
        for _, row in subset.iterrows():
            for reason in row["reason_list"]:
                counts[reason] = counts.get(reason, 0) + 1
        shares[group] = {reason: (count / total if total else 0.0)
                         for reason, count in counts.items()}
    return shares


def _contrast(frame: pd.DataFrame) -> pd.DataFrame:
    shares = _reason_shares(frame)
    reasons = sorted(set(shares["disagreement"]) | set(shares["agreement"]))
    rows = []
    for reason in reasons:
        disagreement = shares["disagreement"].get(reason, 0.0)
        agreement = shares["agreement"].get(reason, 0.0)
        rows.append({
            "reason": reason,
            "share_disagreements": disagreement,
            "share_control": agreement,
            "difference_disagreements_minus_control": disagreement - agreement,
        })
    rows.sort(key=lambda row: -row["difference_disagreements_minus_control"])
    return pd.DataFrame(rows, columns=[
        "reason", "share_disagreements", "share_control",
        "difference_disagreements_minus_control"])


def _examples(frame: pd.DataFrame, per_reason: int = 2) -> pd.DataFrame:
    rows = []
    contrast = _contrast(frame)
    ranked_reasons = contrast["reason"].tolist() if len(contrast) else []
    for group in ("disagreement", "agreement"):
        subset = _explained(frame[frame["group"] == group]).copy()
        subset["rank"] = subset["confidence"].map(CONFIDENCE_RANK).fillna(1)
        subset = subset.sort_values(["rank", "id"], ascending=[False, True])
        for reason in ranked_reasons:
            candidates = subset[subset["reason_list"].apply(lambda values: reason in values)]
            for _, row in candidates.head(per_reason).iterrows():
                text = " ".join(str(row.get("text", "")).split())
                rows.append({
                    "group": group,
                    "reason": reason,
                    "id": row["id"],
                    "reference": row["reference"],
                    "predicted": row["predicted"],
                    "confidence": row["confidence"],
                    "summary": row["summary"],
                    "evidence": row["evidence"],
                    "text_excerpt": text[:240],
                })
    return pd.DataFrame(rows, columns=["group", "reason", "id", "reference",
                                       "predicted", "confidence", "summary",
                                       "evidence", "text_excerpt"])


def write_tables(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame = read_explanations(out_dir)
    selected = _selected_counts(out_dir)

    coverage = _coverage(frame, selected)
    patterns = _patterns(frame, selected)
    cooccurrence = _cooccurrence(frame)
    contrast = _contrast(frame)
    examples = _examples(frame)

    written = {
        "coverage": tables / "q05_coverage.csv",
        "patterns": tables / "q02_patterns.csv",
        "cooccurrence": tables / "q03_cooccurrence.csv",
        "contrast": tables / "q07_reason_contrast.csv",
        "examples": tables / "q04_examples.csv",
    }
    coverage.to_csv(written["coverage"], index=False, float_format="%.6f")
    patterns.to_csv(written["patterns"], index=False, float_format="%.6f")
    cooccurrence.to_csv(written["cooccurrence"], index=False, float_format="%.6f")
    contrast.to_csv(written["contrast"], index=False, float_format="%.6f")
    examples.to_csv(written["examples"], index=False, float_format="%.6f")
    return written


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

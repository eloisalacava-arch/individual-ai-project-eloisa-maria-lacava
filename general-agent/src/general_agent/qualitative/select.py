"""Build the seeded set of disagreements and the control sample of agreements.

Disagreements come from stage 05's ``t08_disagreements.csv`` so the qualitative
set can never drift from the statistical one. Agreements are derived from the
frozen predictions and the sealed reference, using the same ``label_map`` path as
the evaluation. Both draws are seeded and unstratified, and row accounting is
recorded for every restriction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from ..discover import map_labels
from ..io import normalize_ids
from .settings import QualitativeError

SELECTED_COLUMNS = ["group", "id", "reference", "predicted", "ordinal_distance",
                    "severity", "direction", "text"]


def require_frozen(out_dir: Path) -> None:
    if not (out_dir / "results" / "frozen_predictions.csv").exists():
        raise QualitativeError(
            "predictions are not frozen; run stage 05 (evaluate) before the "
            "qualitative analysis")
    if not (out_dir / "results" / "t08_disagreements.csv").exists():
        raise QualitativeError(
            "results/t08_disagreements.csv is missing; run stage 05 (evaluate) first")


def _blind_text(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "blind" / "blind.csv"
    if not path.exists():
        raise QualitativeError(f"{path} is missing; the blind text is required")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    frame["id"] = normalize_ids(frame["id"])
    return frame[["id", "text"]]


def load_disagreements(out_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(out_dir / "results" / "t08_disagreements.csv",
                        dtype="string", keep_default_na=False)
    frame["id"] = normalize_ids(frame["id"])
    frame["group"] = "disagreement"
    return frame


def load_agreements(out_dir: Path, cfg: dict) -> pd.DataFrame:
    """Derive the agreement set from the frozen predictions and sealed reference."""
    frozen = pd.read_csv(out_dir / "results" / "frozen_predictions.csv",
                         dtype="string", keep_default_na=False)
    frozen["id"] = normalize_ids(frozen["id"])
    frozen = frozen.rename(columns={"label": "predicted"})
    sealed = pd.read_csv(out_dir / "sealed" / "benchmark.csv",
                         dtype="string", keep_default_na=False)
    sealed["id"] = normalize_ids(sealed["id"])
    label_column = cfg["benchmark"]["label_column"]
    classes = [entry["name"] for entry in cfg["labels"]["classes"]]
    mapped, unmapped = map_labels(sealed[label_column],
                                  cfg["benchmark"].get("label_map"), classes)
    sealed = sealed.assign(reference=mapped)

    joined = frozen.merge(sealed[["id", "reference"]], on="id", how="inner",
                          validate="one_to_one")
    joined = joined[(joined["status"] == "ok") & joined["reference"].notna()]
    joined = joined[joined["reference"] == joined["predicted"]].copy()
    joined["group"] = "agreement"
    joined["ordinal_distance"] = "0"
    joined["severity"] = "agree"
    joined["direction"] = joined["reference"] + "->" + joined["predicted"]
    if unmapped:
        # Unmapped reference values are never silently classified; they simply
        # cannot be scored, so they are already excluded above.
        pass
    return joined


def _attach_text(frame: pd.DataFrame, text: pd.DataFrame, emit,
                 label: str) -> pd.DataFrame:
    before = len(frame)
    merged = frame.merge(text, on="id", how="left")
    missing = int(merged["text"].isna().sum()) if "text" in merged.columns else before
    if missing:
        merged = merged[merged["text"].notna()].copy()
    if before != len(merged):
        emit(f"{label} without blind text", before, len(merged),
             f"{missing} row(s) have no text to explain")
    return merged


def _select_indices(count: int, n: int, method: str, seed: int) -> list[int]:
    order = list(range(count))
    if method != "head":
        random.Random(seed).shuffle(order)
    if n >= count:
        return order
    return order[:n]


def build_selection(out_dir: Path, cfg: dict, settings: dict) -> tuple[pd.DataFrame, list[dict]]:
    require_frozen(out_dir)
    text = _blind_text(out_dir)
    accounting: list[dict] = []

    def emit(step: str, before: int, after: int, reason: str) -> None:
        accounting.append({
            "step": step, "rows_before": int(before), "rows_after": int(after),
            "rows_dropped": int(before - after), "reason": reason})

    sample = settings["sample"]
    disagreements = _attach_text(load_disagreements(out_dir), text, emit,
                                 "disagreements")
    selected_disagreements = disagreements.iloc[_select_indices(
        len(disagreements), int(sample["n"]), sample.get("method", "random"),
        int(sample["seed"]))].copy()
    emit("disagreements selected", len(disagreements), len(selected_disagreements),
         f"seeded {sample.get('method', 'random')} draw, n={sample['n']}, "
         f"seed={sample['seed']}")

    control_n = int(sample["control_n"])
    if control_n:
        agreements = _attach_text(load_agreements(out_dir, cfg), text, emit,
                                  "agreements")
        selected_agreements = agreements.iloc[_select_indices(
            len(agreements), control_n, sample.get("method", "random"),
            int(sample["seed"]) + 1)].copy()
    else:
        agreements = pd.DataFrame(columns=SELECTED_COLUMNS)
        selected_agreements = pd.DataFrame(columns=SELECTED_COLUMNS)
    emit("agreement control selected", len(agreements), len(selected_agreements),
         f"seeded {sample.get('method', 'random')} draw, control_n={control_n}, "
         f"seed={int(sample['seed']) + 1}")

    selected = pd.concat([selected_disagreements, selected_agreements],
                         ignore_index=True)
    for column in SELECTED_COLUMNS:
        if column not in selected.columns:
            selected[column] = ""
    return selected[SELECTED_COLUMNS], accounting


def selection_signature(settings: dict) -> str:
    sample = settings["sample"]
    core = {
        "study": settings["_meta"]["study_config_sha256"],
        "n": int(sample["n"]),
        "control_n": int(sample["control_n"]),
        "method": sample.get("method", "random"),
        "seed": int(sample["seed"]),
        "vocabulary_version": int(settings.get("vocabulary_version", 1)),
        "prompt_version": int(settings.get("prompt_version", 1)),
    }
    return json.dumps(core, sort_keys=True)


def write_selection(out_dir: Path, selected: pd.DataFrame,
                    accounting: list[dict], settings: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "selected.csv", index=False)
    pd.DataFrame(accounting).to_csv(tables / "q00_selection.csv", index=False)
    (out_dir / "selection_meta.json").write_text(
        json.dumps({
            "signature": selection_signature(settings),
            "n_selected": int(len(selected)),
            "n_disagreements": int((selected["group"] == "disagreement").sum()),
            "n_agreements": int((selected["group"] == "agreement").sum()),
            "seed": int(settings["sample"]["seed"]),
            "accounting": accounting,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_dir / "selected.csv"


def load_selection(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "selected.csv"
    if not path.exists():
        raise QualitativeError("selected.csv is missing; run the pilot first")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    return frame


def stored_signature(out_dir: Path) -> str | None:
    path = out_dir / "selection_meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("signature")
    except (json.JSONDecodeError, OSError):
        return None

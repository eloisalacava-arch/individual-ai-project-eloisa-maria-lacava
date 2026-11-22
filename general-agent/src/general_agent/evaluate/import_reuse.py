"""Materialise a reused prediction file into the canonical predictions schema.

A study may reuse predictions produced elsewhere, so the same evaluation code
can score them without a live run. This is generic: any
config with ``dataset.predictions.reuse: true`` goes through the same path.

Nothing is overwritten: if ``predictions.csv`` already exists (a live run), this
function does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import class_names
from ..io import normalize_ids, read_table
from ..provenance import file_sha256, now_iso


class ReuseError(RuntimeError):
    """Raised when a reuse source cannot be turned into canonical predictions."""


def materialise_predictions(out_dir: Path, cfg: dict) -> dict | None:
    canonical = out_dir / "predictions.csv"
    if canonical.exists():
        return None

    predictions_cfg = cfg["dataset"].get("predictions")
    if not predictions_cfg or not predictions_cfg.get("reuse"):
        return None

    source = Path(predictions_cfg["path"])
    frame = read_table(source, predictions_cfg["format"],
                       predictions_cfg["delimiter"], predictions_cfg["encoding"])
    id_column = (predictions_cfg.get("id_column")
                 or cfg["dataset"].get("id_column"))
    label_column = predictions_cfg["label_column"]
    if not id_column or id_column not in frame.columns:
        raise ReuseError(
            f"reused predictions need an ID column; got {id_column!r} in {list(frame.columns)}"
        )
    if label_column not in frame.columns:
        raise ReuseError(f"reused predictions have no {label_column!r} column")

    status_column = predictions_cfg.get("status_column")
    model_column = predictions_cfg.get("model_column")

    rows = []
    for _, record in frame.iterrows():
        label = str(record[label_column]).strip()
        status = (str(record[status_column]).strip()
                  if status_column and status_column in frame.columns else "ok")
        model = (str(record[model_column]).strip()
                 if model_column and model_column in frame.columns else cfg["model"]["name"])
        rows.append({
            "id": str(record[id_column]).strip(),
            "label": label,
            "status": status,
            "attempts": 1,
            "model": model,
        })

    predictions = pd.DataFrame(rows, columns=["id", "label", "status", "attempts", "model"])
    predictions["id"] = normalize_ids(predictions["id"])
    if predictions["id"].duplicated().any():
        raise ReuseError("reused predictions contain duplicate IDs")

    classes = class_names(cfg)
    ok_labels = predictions.loc[predictions["status"] == "ok", "label"]
    unknown = sorted(set(ok_labels) - set(classes))
    if unknown:
        raise ReuseError(f"reused predictions contain undeclared labels: {unknown}")

    canonical.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(canonical, index=False)

    info = {
        "imported_at": now_iso(),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "canonical_path": str(canonical),
        "canonical_sha256": file_sha256(canonical),
        "n_rows": int(len(predictions)),
        "status_counts": {str(k): int(v)
                          for k, v in predictions["status"].value_counts().items()},
        "models": sorted(set(predictions["model"])),
        "note": "reused predictions; zero model calls in this stage",
    }
    (out_dir / "predictions_import.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info
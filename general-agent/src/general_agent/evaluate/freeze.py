"""Freeze predictions before any benchmark access.

The freeze is a copy plus a hash, not a permission change: the original
``predictions.csv`` stays exactly where the classifier wrote it. The point is
that the evaluation reads the frozen copy, so a later edit to the working file
cannot silently change what was scored.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd

from ..provenance import file_sha256, now_iso

PRED_COLUMNS = ["id", "label", "status", "attempts", "model"]


class FreezeError(RuntimeError):
    """Raised when the prediction file is not in a scorable state."""


def freeze_predictions(out_dir: Path, results_dir: Path,
                       classes: list[str]) -> dict:
    source = out_dir / "predictions.csv"
    if not source.exists():
        raise FreezeError(f"{source} is missing: run classification first")

    frame = pd.read_csv(source, dtype="string", keep_default_na=False)
    if list(frame.columns) != PRED_COLUMNS:
        raise FreezeError(
            f"predictions columns are {list(frame.columns)}, expected {PRED_COLUMNS}"
        )
    if frame["id"].str.strip().eq("").any():
        raise FreezeError("predictions contain a blank ID")
    if frame["id"].duplicated().any():
        raise FreezeError("predictions contain duplicate IDs")

    status_counts = frame["status"].value_counts().to_dict()
    label_counts = frame["label"].value_counts().to_dict()
    models = sorted(set(frame["model"].dropna()))
    trimmed = frame["label"].str.strip()
    non_empty = trimmed[trimmed != ""]
    unknown = sorted(set(non_empty) - set(classes))
    ok_without_label = int(((frame["status"] == "ok") & (trimmed == "")).sum())

    results_dir.mkdir(parents=True, exist_ok=True)
    frozen = results_dir / "frozen_predictions.csv"
    shutil.copy2(source, frozen)

    info = {
        "frozen_at": now_iso(),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "frozen_path": str(frozen),
        "frozen_sha256": file_sha256(frozen),
        "n_rows": int(len(frame)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "predicted_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "models": models,
        "labels_outside_declared_classes": unknown,
        "ok_rows_without_label": ok_without_label,
        "n_not_ok": int((frame["status"] != "ok").sum()),
        "n_scorable": int((frame["status"] == "ok").sum()),
        "note": "evaluation reads frozen_predictions.csv; the working predictions.csv is untouched",
    }
    (results_dir / "predictions_freeze.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if unknown:
        raise FreezeError(
            f"predictions contain labels outside the declared classes: {unknown}"
        )
    if ok_without_label:
        raise FreezeError(
            f"{ok_without_label} row(s) have status 'ok' but no label; refusing to score"
        )
    return info
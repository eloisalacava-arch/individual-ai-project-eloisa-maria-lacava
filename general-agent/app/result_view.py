"""Read the engine's result tables for display and download.

Nothing here recomputes a metric. Every number shown in the interface comes from
a CSV under ``runs/<run_id>/results/`` written by stage 05 or stage 06, or from
``report/report.md`` written by stage 07.
"""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import pandas as pd

RESULT_TABLES = [
    "t01_join_and_sample.csv",
    "t02_accuracy.csv",
    "t03_kappa.csv",
    "t04_confusion_matrix.csv",
    "t05_per_class_metrics.csv",
    "t06_class_distribution.csv",
    "t07_ordinal_errors.csv",
    "t08_disagreements.csv",
    "t09_category_summary.csv",
    "t10_error_status.csv",
    "t11_disagreement_pairs.csv",
    "t12_text_profile.csv",
    "t12b_length_test.csv",
    "t13_reference_agreement.csv",
]


def fmt(value, digits: int = 3) -> str:
    """Format a metric for display; degenerate values are labelled, not zeroed."""
    if value is None:
        return "not estimable"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "not estimable"
    return f"{number:.{digits}f}"


def read_text(path: Path) -> str:
    if not Path(path).exists():
        return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def load_results(run_dir: Path) -> dict[str, pd.DataFrame]:
    results_dir = Path(run_dir) / "results"
    frames: dict[str, pd.DataFrame] = {}
    for name in RESULT_TABLES:
        path = results_dir / name
        if path.exists():
            frames[name] = pd.read_csv(path)
    return frames


def accuracy_row(frames: dict[str, pd.DataFrame]) -> dict:
    frame = frames.get("t02_accuracy.csv")
    if frame is None or frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def kappa_rows(frames: dict[str, pd.DataFrame]) -> list[dict]:
    frame = frames.get("t03_kappa.csv")
    if frame is None:
        return []
    return frame.to_dict(orient="records")


def confusion_frames(run_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    path = Path(run_dir) / "results" / "t04_confusion_matrix.csv"
    if not path.exists():
        return None, None
    full = pd.read_csv(path, index_col=0)
    matrix = full.drop(index="total_predicted", errors="ignore")
    matrix = matrix.drop(columns="total_human", errors="ignore")
    return matrix, full


def per_class_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    return frames.get("t05_per_class_metrics.csv")


def distribution_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    return frames.get("t06_class_distribution.csv")


def join_sample_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    return frames.get("t01_join_and_sample.csv")


def text_disagreements(run_dir: Path) -> pd.DataFrame | None:
    """Stage 05 disagreements joined to the blind text for reading."""
    run_dir = Path(run_dir)
    path = run_dir / "results" / "t08_disagreements.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    blind_path = run_dir / "blind" / "blind.csv"
    if blind_path.exists():
        blind = pd.read_csv(blind_path, dtype="string", keep_default_na=False)
        frame = frame.merge(blind, on="id", how="left")
    columns = [c for c in ["id", "reference", "predicted", "severity",
                           "direction", "text"] if c in frame.columns]
    return frame[columns] if columns else frame


def prediction_rows(run_dir: Path) -> pd.DataFrame | None:
    run_dir = Path(run_dir)
    for candidate in (run_dir / "results" / "frozen_predictions.csv",
                      run_dir / "predictions.csv"):
        if candidate.exists():
            return pd.read_csv(candidate, dtype="string", keep_default_na=False)
    return None


def predictions_table(run_dir: Path, blind_path: Path | None = None,
                      limit: int = 200) -> pd.DataFrame | None:
    """Predictions with a short text excerpt; no reference label is present."""
    frame = prediction_rows(run_dir)
    if frame is None:
        return None
    frame = frame.copy()
    blind_path = blind_path or Path(run_dir) / "blind" / "blind.csv"
    if blind_path.exists():
        blind = pd.read_csv(blind_path, dtype="string", keep_default_na=False)
        frame = frame.merge(blind, on="id", how="left")
    if "text" in frame.columns:
        frame["text"] = frame["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 90)
    if "model" in frame.columns:
        frame = frame.drop(columns="model")
    return frame.head(limit)


def read_bytes(path: Path) -> bytes | None:
    path = Path(path)
    return path.read_bytes() if path.exists() else None


def make_zip(base: Path, files: list[Path]) -> bytes:
    """Zip a list of files, storing paths relative to ``base`` when possible."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            path = Path(path)
            if not path.exists() or not path.is_file():
                continue
            try:
                name = str(path.relative_to(base))
            except ValueError:
                name = path.name
            archive.write(path, arcname=name)
    return buffer.getvalue()


QUALITATIVE_TABLES = [
    "q00_selection.csv", "q02_patterns.csv", "q03_cooccurrence.csv",
    "q04_examples.csv", "q05_coverage.csv", "q07_reason_contrast.csv",
]


def qualitative_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "qualitative"


def load_qualitative(run_dir: Path) -> dict[str, pd.DataFrame]:
    tables = qualitative_dir(run_dir) / "tables"
    frames: dict[str, pd.DataFrame] = {}
    for name in QUALITATIVE_TABLES:
        path = tables / name
        if not path.exists():
            continue
        try:
            frames[name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
    return frames


def qualitative_explanations(run_dir: Path) -> pd.DataFrame | None:
    path = qualitative_dir(run_dir) / "explanations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def qualitative_pilot_explanations(run_dir: Path) -> pd.DataFrame | None:
    path = qualitative_dir(run_dir) / "pilot" / "pilot_explanations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def qualitative_narrative(run_dir: Path) -> str:
    return read_text(qualitative_dir(run_dir) / "narrative_report.md")


def qualitative_stage(run_dir: Path) -> dict:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return {}
    import json
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return (manifest.get("stages") or {}).get("08_qualitative", {})


def qualitative_calls_used(run_dir: Path) -> int:
    directory = qualitative_dir(run_dir)
    total = 0
    for name in ("pilot/pilot_attempts.jsonl", "attempts.jsonl"):
        path = directory / name
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
    return total


def qualitative_files(run_dir: Path) -> list[Path]:
    directory = qualitative_dir(run_dir)
    if not directory.exists():
        return []
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "quarantine" not in path.parts:
            files.append(path)
    return files


def result_files(run_dir: Path) -> list[Path]:
    results_dir = Path(run_dir) / "results"
    if not results_dir.exists():
        return []
    return sorted(path for path in results_dir.iterdir() if path.is_file())


def provenance_files(run_dir: Path) -> list[Path]:
    """Everything needed to audit a run, excluding the sealed benchmark."""
    run_dir = Path(run_dir)
    files: list[Path] = []
    for name in ("study.yaml", "manifest.json", "decisions.json", "audit.txt"):
        candidate = run_dir / name
        if candidate.exists():
            files.append(candidate)
    for folder in ("blind", "pilot", "logs", "results", "report"):
        directory = run_dir / folder
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file():
                files.append(path)
    return files

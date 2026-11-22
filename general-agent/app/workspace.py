"""Run workspaces, upload staging, and table inspection.

A study is one self-contained directory under ``runs/<run_id>/``: the generated
``study.yaml``, unmodified copies of the uploads in ``inputs/``, and the engine
artefacts the stages write. Nothing here modifies a configured input.

This module deliberately avoids importing Streamlit so it can be unit-tested.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from general_agent.io import detect_format, read_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
DRAFTS_ROOT = RUNS_ROOT / "_drafts"
REGISTRY_PATH = PROJECT_ROOT / "configs" / "ui_providers.yaml"

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".txt", ".jsonl", ".ndjson", ".parquet", ".pq")
_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (name or "").strip()).strip("-._")
    return (slug or "study")[:60]


def next_run_id(name: str) -> str:
    """A stable, unique run id. An existing directory is never reused."""
    base = slugify(name)
    if not (RUNS_ROOT / base).exists():
        return base
    counter = 2
    while (RUNS_ROOT / f"{base}-{counter}").exists():
        counter += 1
    return f"{base}-{counter}"


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def display_path(path: Path) -> str:
    """A short path for the interface: relative to the project when possible."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Upload staging                                                              #
# --------------------------------------------------------------------------- #

def staging_dir(session_id: str) -> Path:
    path = DRAFTS_ROOT / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_upload(session_id: str, role: str, filename: str, data: bytes) -> dict:
    """Write an upload to the session draft area and describe it.

    The file is written under a fixed name per role so repeated uploads do not
    accumulate. The original filename and digest are kept for provenance.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        suffix = ".csv"
    target = staging_dir(session_id) / f"{role}{suffix}"
    target.write_bytes(data)
    return {
        "path": str(target),
        "original_name": filename,
        "suffix": suffix,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _detect_encoding(path: Path) -> str:
    for encoding in _ENCODINGS:
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _detect_delimiter(path: Path, encoding: str) -> str:
    try:
        sample = path.open(encoding=encoding, errors="replace").read(16384)
    except OSError:
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return "\t" if path.suffix.lower() == ".tsv" else ","


def inspect_table(path: str | Path, delimiter: str | None = None,
                  encoding: str | None = None) -> dict:
    """Read a table and return everything the UI needs to offer column choices."""
    path = Path(path)
    fmt = detect_format(path, "auto")
    encoding = encoding or _detect_encoding(path)
    delimiter = delimiter or (_detect_delimiter(path, encoding) if fmt == "csv" else ",")
    frame = read_table(path, fmt, delimiter, encoding)
    return {
        "path": str(path),
        "format": fmt,
        "delimiter": delimiter,
        "encoding": encoding,
        "columns": [str(column) for column in frame.columns],
        "n_rows": int(len(frame)),
        "preview": frame.head(5),
        "frame": frame,
    }


# --------------------------------------------------------------------------- #
# Run inputs                                                                  #
# --------------------------------------------------------------------------- #

def materialise_inputs(run_dir_path: Path, dataset_staged: dict,
                       benchmark_staged: dict, dataset_meta: dict,
                       make_id: bool = False) -> dict:
    """Copy the uploads into the run and, optionally, add a row-number ID.

    No row is ever dropped. Adding an ID writes a derived copy and is reported
    through ``row_note`` so the interface can print rows before -> after.
    """
    inputs = run_dir_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    dataset_suffix = Path(dataset_staged["path"]).suffix.lower()
    benchmark_suffix = Path(benchmark_staged["path"]).suffix.lower()

    if make_id:
        derived = inputs / "dataset_with_id.csv"
        frame = pd.DataFrame(dataset_meta["frame"]).copy()
        frame.insert(0, "observation_id", [f"obs-{i + 1}" for i in range(len(frame))])
        frame.to_csv(derived, index=False)
        dataset_rel = "inputs/dataset_with_id.csv"
        dataset_format, delimiter, encoding = "csv", ",", "utf-8"
        dataset_id_column = "observation_id"
        row_note = (f"rows: {len(frame)} -> {len(frame)} "
                    "(added observation_id from row numbers; no rows dropped)")
    else:
        target = inputs / f"dataset{dataset_suffix}"
        shutil.copy2(dataset_staged["path"], target)
        dataset_rel = f"inputs/dataset{dataset_suffix}"
        dataset_format = detect_format(target, "auto")
        delimiter = dataset_meta.get("delimiter", ",")
        encoding = dataset_meta.get("encoding", "utf-8")
        dataset_id_column = None
        row_note = ""

    benchmark_target = inputs / f"benchmark{benchmark_suffix}"
    shutil.copy2(benchmark_staged["path"], benchmark_target)
    benchmark_rel = f"inputs/benchmark{benchmark_suffix}"

    return {
        "dataset_rel": dataset_rel,
        "benchmark_rel": benchmark_rel,
        "dataset_format": dataset_format,
        "dataset_delimiter": delimiter,
        "dataset_encoding": encoding,
        "dataset_id_column": dataset_id_column,
        "row_note": row_note,
    }


# --------------------------------------------------------------------------- #
# Run discovery and state                                                     #
# --------------------------------------------------------------------------- #

def load_manifest(run_id: str) -> dict:
    path = run_dir(run_id) / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def read_decisions(run_id: str) -> dict:
    path = run_dir(run_id) / "decisions.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def predictions_counts(run_id: str) -> dict:
    path = run_dir(run_id) / "predictions.csv"
    if not path.exists():
        return {"total": 0, "ok": 0, "pending": 0, "failed": 0, "statuses": {}}
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    statuses = frame["status"].value_counts().to_dict()
    total = len(frame)
    ok = int(statuses.get("ok", 0))
    return {
        "total": total,
        "ok": ok,
        "pending": int(statuses.get("pending", 0)),
        "failed": total - ok - int(statuses.get("pending", 0)),
        "statuses": {str(key): int(value) for key, value in statuses.items()},
    }


def authorization_path(run_id: str) -> Path:
    return run_dir(run_id) / "ui_authorization.json"


def write_authorization(run_id: str, payload: dict) -> Path:
    from general_agent.io import write_json_atomic
    return write_json_atomic(authorization_path(run_id), payload)


def read_authorization(run_id: str) -> dict | None:
    path = authorization_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def progress(run_id: str) -> dict:
    """Everything the simple Analyze screen needs, read from disk only."""
    flags = stage_flags(run_id)
    counts = flags.pop("counts", {"total": 0, "ok": 0, "pending": 0, "failed": 0,
                                  "statuses": {}})
    directory = run_dir(run_id)
    qualitative_explanations = directory / "qualitative" / "explanations.csv"
    qualitative = {"started": qualitative_explanations.exists(),
                   "total": 0, "ok": 0, "pending": 0}
    if qualitative_explanations.exists():
        frame = pd.read_csv(qualitative_explanations, dtype="string",
                            keep_default_na=False)
        qualitative = {
            "started": True,
            "total": int(len(frame)),
            "ok": int((frame["status"] == "ok").sum()),
            "pending": int((frame["status"] != "ok").sum()),
        }
    disagreements = 0
    t08 = directory / "results" / "t08_disagreements.csv"
    if t08.exists():
        disagreements = int(len(pd.read_csv(t08, dtype="string")))
    try:
        from general_agent.report import paths as report_paths
        study_report = (report_paths.report_dir(run_id) / "study_report.md").exists()
    except Exception:
        study_report = False
    return {
        "flags": flags,
        "counts": counts,
        "calls_used": calls_used(run_id),
        "cost": cost_totals(run_id),
        "qualitative": qualitative,
        "disagreements": disagreements,
        "authorized": authorization_path(run_id).exists(),
        "study_report": study_report,
    }


def attempt_log_paths(run_id: str) -> list[Path]:
    directory = run_dir(run_id)
    return [directory / "pilot" / "pilot_attempts.jsonl",
            directory / "logs" / "attempts.jsonl"]


def calls_used(run_id: str) -> int:
    """Attempts recorded so far, pilot and full together."""
    total = 0
    for path in attempt_log_paths(run_id):
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
    return total


def cost_totals(run_id: str) -> dict:
    """Provider-reported cost and the character-based floor from the logs."""
    actual = 0.0
    estimated = 0.0
    reported = 0
    for path in attempt_log_paths(run_id):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cost = record.get("cost_usd")
            if isinstance(cost, (int, float)):
                actual += float(cost)
                reported += 1
            estimate = record.get("estimated_cost_usd")
            if isinstance(estimate, (int, float)):
                estimated += float(estimate)
    return {"actual_usd": actual, "reported_calls": reported,
            "estimated_usd": estimated}


def sealed_marker(run_id: str) -> Path:
    """A seal marker that is never locked during classification.

    ``sealed/benchmark.csv`` lives inside the directory the engine chmods to
    ``000`` while a run is in progress, so ``Path.exists()`` returns ``False``
    for it during classification. ``blind_public_info.json`` is written at seal
    time at the run root and stays readable, so it is the reliable signal.
    """
    return run_dir(run_id) / "blind_public_info.json"


def stage_flags(run_id: str) -> dict:
    directory = run_dir(run_id)
    decisions = read_decisions(run_id)
    counts = predictions_counts(run_id)
    return {
        "configured": (directory / "study.yaml").exists(),
        "audited": bool(decisions) and decisions.get("status") == "resolved",
        "sealed": (directory / "blind" / "blind.csv").exists()
        and sealed_marker(run_id).exists(),
        "piloted": (directory / "pilot" / "pilot_predictions.csv").exists(),
        "approved": (directory / "pilot" / "approval.json").exists(),
        "classified": counts["total"] > 0 and counts["pending"] == 0,
        "frozen": (directory / "results" / "frozen_predictions.csv").exists(),
        "evaluated": (directory / "results" / "t02_accuracy.csv").exists(),
        "reported": (directory / "report" / "report.md").exists(),
        "counts": counts,
    }


def list_runs() -> list[dict]:
    """Every UI study, newest first. Drafts and case studies are excluded."""
    if not RUNS_ROOT.exists():
        return []
    runs = []
    for directory in RUNS_ROOT.iterdir():
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        config_path = directory / "study.yaml"
        if not config_path.exists():
            continue
        manifest = load_manifest(directory.name)
        title = directory.name
        try:
            from general_agent.config import load_config
            title = load_config(config_path).get("report", {}).get("title") or title
        except Exception:  # a drifted config must not break the run list
            pass
        runs.append({
            "run_id": directory.name,
            "title": title,
            "updated_at": manifest.get("updated_at") or manifest.get("created_at") or "",
            "flags": stage_flags(directory.name),
        })
    runs.sort(key=lambda item: item["updated_at"], reverse=True)
    return runs

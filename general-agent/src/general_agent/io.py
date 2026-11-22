"""Table and file I/O helpers.

Rules enforced here rather than in each script:
  * a missing input file fails with a readable message, never an empty frame;
  * IDs are normalised to trimmed strings so ``123`` and ``"123"`` join;
  * writes are atomic (temp file + rename) so an interrupted run cannot leave
    a half-written manifest or decisions file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

SUPPORTED_FORMATS = ("auto", "csv", "parquet", "jsonl")

FORMAT_BY_SUFFIX = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_format(path: str | Path, fmt: str = "auto") -> str:
    if fmt != "auto":
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"unsupported table format: {fmt!r}")
        return fmt
    suffix = Path(path).suffix.lower()
    if suffix not in FORMAT_BY_SUFFIX:
        raise ValueError(
            f"cannot infer table format from {Path(path).name!r}; "
            f"set format explicitly to one of {SUPPORTED_FORMATS[1:]}"
        )
    return FORMAT_BY_SUFFIX[suffix]


def _check_duplicate_columns(path: Path, delimiter: str, encoding: str) -> None:
    """pandas silently renames duplicate headers; catch that before it corrupts a join."""
    with path.open(newline="", encoding=encoding) as fh:
        header = next(csv.reader(fh, delimiter=delimiter), [])
    seen: dict[str, int] = {}
    for name in header:
        seen[name] = seen.get(name, 0) + 1
    duplicates = [name for name, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"{path.name}: duplicate column names {duplicates}")


def read_table(
    path: str | Path,
    fmt: str = "auto",
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Check the path in the config; "
            "paths resolve relative to the config file's directory."
        )
    resolved = detect_format(path, fmt)
    if resolved == "csv":
        _check_duplicate_columns(path, delimiter, encoding)
        return pd.read_csv(path, sep=delimiter, encoding=encoding)
    if resolved == "parquet":
        return pd.read_parquet(path)
    records = []
    with path.open(encoding=encoding) as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {number}: invalid JSONL ({exc})") from exc
    return pd.DataFrame.from_records(records)


def normalize_ids(series: pd.Series) -> pd.Series:
    """Trim IDs and render integral floats without a spurious ``.0``.

    1 and "1" become "1"; 1.0 becomes "1"; NA stays NA so null IDs are visible
    rather than silently becoming the string "nan".
    """
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if len(non_null) and (non_null == non_null.round()).all():
            series = series.round().astype("Int64")
    return series.astype("string").str.strip()


def write_text_atomic(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_json_atomic(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def file_fingerprint(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
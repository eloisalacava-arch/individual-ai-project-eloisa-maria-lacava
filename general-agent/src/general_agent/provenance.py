"""Provenance helpers shared by stages 02-04.

All timestamps are UTC ISO-8601. Manifest updates are merge-not-overwrite so
each stage appends its own section instead of replacing the run record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .io import write_json_atomic


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(out_dir: str | Path) -> dict:
    path = Path(out_dir) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(out_dir: str | Path, section: str, payload: dict) -> dict:
    """Merge ``payload`` under ``section`` and persist the manifest."""
    manifest = load_manifest(out_dir)
    manifest.setdefault("stages", {})[section] = payload
    manifest["updated_at"] = now_iso()
    write_json_atomic(Path(out_dir) / "manifest.json", manifest)
    return manifest


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
"""Shared test helpers: build a valid study config on a temporary path."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from general_agent.audit import run_audit

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def make_config(
    tmp_path: Path,
    dataset: str = "dataset_clean.csv",
    benchmark: str = "benchmark_clean.csv",
    patch: dict | None = None,
    name: str = "study.yaml",
) -> Path:
    cfg = {
        "run": {
            "run_id": "test_run",
            "seed": 20261116,
            "output_root": str(tmp_path / "runs"),
        },
        "dataset": {
            "path": str(FIXTURES / dataset),
            "format": "csv",
            "id_column": "review_id",
            "text_column": "review_text",
        },
        "benchmark": {
            "path": str(FIXTURES / benchmark),
            "format": "csv",
            "id_column": "review_id",
            "label_column": "sentiment",
        },
        "labels": {
            "classes": [
                {"name": "positive", "definition": "favourable overall stance"},
                {"name": "negative", "definition": "unfavourable overall stance"},
            ]
        },
        "task": {"role": "a test classifier", "output_contract": "ANSWER: {class}"},
        "model": {"provider": "opencode_cli", "name": "test/model"},
        "budget": {"max_calls": 10},
    }
    if patch:
        _merge(cfg, patch)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path

def write_rows(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def prepare_case(
    tmp_path: Path,
    n: int = 12,
    sample_n: int | None = None,
    pilot_n: int = 3,
    max_calls: int = 40,
    duplicate_pairs: int = 0,
    marker: str | None = None,
    subgroup_columns: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Create a small dataset/benchmark pair and run the audit.

    ``marker`` is written into a benchmark-only column so leakage tests can
    assert it never reaches a prompt or prediction file.
    """
    dataset_path = tmp_path / "source_dataset.csv"
    benchmark_path = tmp_path / "source_benchmark.csv"

    texts = [
        f"Observation {index}: the product worked reliably and support answered quickly."
        for index in range(n)
    ]
    # Make some texts identical without changing the row count.
    for index in range(duplicate_pairs):
        mirrored = n - 1 - index
        if mirrored > index:
            texts[mirrored] = texts[index]

    dataset_rows = [[f"r{index}", texts[index]] for index in range(n)]
    benchmark_rows = []
    for index in range(n):
        label = "positive" if index % 2 == 0 else "negative"
        row = [f"r{index}", label]
        if marker is not None:
            row.append(marker)
        benchmark_rows.append(row)

    write_rows(dataset_path, ["review_id", "review_text"], dataset_rows)
    header = ["review_id", "sentiment"]
    if marker is not None:
        header.append("human_notes")
    write_rows(benchmark_path, header, benchmark_rows)

    patch = {
        "dataset": {"path": str(dataset_path)},
        "benchmark": {"path": str(benchmark_path)},
        "sample": {"n": sample_n},
        "pilot": {"n": pilot_n},
        "budget": {"max_calls": max_calls, "max_usd": None},
    }
    if subgroup_columns:
        patch["evaluation"] = {"subgroup_columns": subgroup_columns}
    config_path = make_config(tmp_path, patch=patch)
    code = run_audit(config_path)
    if code != 0:
        raise AssertionError(f"audit failed with code {code}")
    return config_path, dataset_path, benchmark_path

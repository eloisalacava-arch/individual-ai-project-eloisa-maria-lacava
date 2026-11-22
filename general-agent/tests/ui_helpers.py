"""Build a complete UI study on a temporary path, without Streamlit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config_builder import build_config, write_config

DEFAULT_MODEL = {
    "provider": "opencode_cli",
    "name": "test/model",
    "temperature": None,
    "timeout_s": 180,
    "max_attempts": 3,
    "concurrency": 1,
}


def class_names(classes: int) -> list[str]:
    if classes == 3:
        return ["negative", "neutral", "positive"]
    return ["positive", "negative"]


def make_ui_study(tmp_path: Path, run_id: str = "ui_study", n: int = 12,
                  classes: int = 2, max_calls: int = 40, marker: str | None = None,
                  subgroup_columns: list[str] | None = None, seed: int = 20261116,
                  sample_n: int | None = None, on_unknown_label: str = "error",
                  on_id_mismatch: str = "error",
                  label_map: dict | None = None,
                  ordered: bool | None = None,
                  reference_description: str = "synthetic reference") -> dict:
    run_dir = tmp_path / "runs" / run_id
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    names = class_names(classes)
    rows = [[f"r{i}", f"observation {i} with enough words to look like real text"]
            for i in range(n)]
    bench = []
    for i in range(n):
        row = [f"r{i}", names[i % len(names)]]
        if marker is not None:
            row.append(marker)
        bench.append(row)

    dataset = inputs / "dataset.csv"
    benchmark = inputs / "benchmark.csv"
    pd.DataFrame(rows, columns=["review_id", "review_text"]).to_csv(dataset, index=False)
    header = ["review_id", "sentiment"] + (["human_notes"] if marker is not None else [])
    pd.DataFrame(bench, columns=header).to_csv(benchmark, index=False)

    spec = {
        "run_id": run_id,
        "dataset_rel": "inputs/dataset.csv",
        "dataset_format": "csv",
        "dataset_delimiter": ",",
        "dataset_encoding": "utf-8",
        "dataset_id_column": "review_id",
        "text_column": "review_text",
        "benchmark_rel": "inputs/benchmark.csv",
        "benchmark_format": "csv",
        "benchmark_delimiter": ",",
        "benchmark_encoding": "utf-8",
        "benchmark_id_column": "review_id",
        "label_column": "sentiment",
        "label_map": label_map or {},
        "on_unknown_label": on_unknown_label,
        "on_id_mismatch": on_id_mismatch,
        "classes": [{"name": name, "definition": f"definition of {name}"} for name in names],
        "ordered": (classes == 3) if ordered is None else ordered,
        "rows_mode": "subset" if sample_n else "all",
        "sample_n": sample_n,
        "seed": seed,
        "task_role": "a test classifier",
        "decision_rule": "",
        "model": dict(DEFAULT_MODEL),
        "max_calls": max_calls,
        "max_usd": 1.0,
        "price_per_million": {"input": 0.0, "output": 0.0},
        "report_title": run_id,
        "report_author": "test",
        "task_description": "synthetic task",
        "reference_description": reference_description,
        "subgroup_columns": subgroup_columns or [],
    }
    config_path = write_config(run_dir, build_config(spec))
    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "dataset": dataset,
        "benchmark": benchmark,
        "classes": names,
        "spec": spec,
    }

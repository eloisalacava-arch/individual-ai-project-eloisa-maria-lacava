"""Turn interface choices into a normal engine study configuration.

Every field is written explicitly, so the manifest's ``defaults_applied`` is
empty and a reader can tell a user choice from an inherited setting. The result
is validated with the engine's own ``load_config`` before it is saved.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from general_agent.config import ConfigError, load_config

from .definitions import is_placeholder_definition

PILOT_N = 5

DEFAULT_LIMITATIONS = [
    "Human labels are a benchmark, not adjudicated truth; a disagreement is not a confirmed model error.",
    "The sample is not stratified; class prevalence reflects the uploaded file.",
    "Model temperature is not controlled; repeated runs may differ.",
    "Rows are assumed independent; duplicate observations may inflate apparent precision.",
]


def suggest_max_calls(rows: int, pilot_n: int = PILOT_N) -> int:
    """Minimum calls plus a retry buffer. The engine counts every attempt."""
    rows = max(int(rows), 0)
    buffer = max(5, math.ceil(rows * 0.1))
    return pilot_n + rows + buffer


def build_config(spec: dict) -> dict:
    classes = [
        {"name": entry["name"].strip(), "definition": entry["definition"].strip()}
        for entry in spec["classes"]
        if entry.get("name", "").strip()
    ]
    ordered = bool(spec.get("ordered"))
    kappa = ["unweighted"] + (["quadratic"] if ordered else [])

    sample_n = spec.get("sample_n") if spec.get("rows_mode") == "subset" else None
    if sample_n is not None:
        sample_n = int(sample_n)

    label_map = {
        str(key): str(value)
        for key, value in (spec.get("label_map") or {}).items()
        if value
    }

    dataset = {
        "path": spec["dataset_rel"],
        "format": spec.get("dataset_format", "auto"),
        "delimiter": spec.get("dataset_delimiter", ","),
        "encoding": spec.get("dataset_encoding", "utf-8"),
        "id_column": spec["dataset_id_column"],
        "text_column": spec["text_column"],
    }
    benchmark = {
        "path": spec["benchmark_rel"],
        "format": spec.get("benchmark_format", "auto"),
        "delimiter": spec.get("benchmark_delimiter", ","),
        "encoding": spec.get("benchmark_encoding", "utf-8"),
        "id_column": spec["benchmark_id_column"],
        "label_column": spec["label_column"],
        "extra_label_columns": [],
        "reference_labels": {},
        "on_unknown_label": spec.get("on_unknown_label", "error"),
        "on_id_mismatch": spec.get("on_id_mismatch", "error"),
        "allow_duplicate_ids": False,
    }
    if label_map:
        benchmark["label_map"] = label_map

    config = {
        "run": {
            "run_id": spec["run_id"],
            "seed": int(spec.get("seed", 20261116)),
            "output_root": "..",
        },
        "dataset": dataset,
        "benchmark": benchmark,
        "labels": {
            "mode": "single_label",
            "ordered": ordered,
            "classes": classes,
        },
        "sample": {
            "n": sample_n,
            "unique_text": True,
            "method": "random",
        },
        "pilot": {"n": PILOT_N},
        "task": {
            "role": spec.get("task_role") or "a classifier for short observations",
            "forbid_outside_knowledge": True,
            "output_contract": "ANSWER: {class}",
        },
        "model": spec["model"],
        "budget": {
            "max_calls": int(spec["max_calls"]),
            "max_usd": spec.get("max_usd"),
            "price_per_million": spec.get("price_per_million")
            or {"input": 0.0, "output": 0.0},
        },
        "evaluation": {
            "ci_level": 0.95,
            "bootstrap_draws": 2000,
            "kappa": kappa,
            "abstention_policy": ["exclude", "count_as_error"],
            "subgroup_columns": list(spec.get("subgroup_columns") or []),
            "subgroup_reference": None,
            "subgroup_composites": {},
        },
        "report": {
            "title": spec.get("report_title") or spec["run_id"],
            "author": spec.get("report_author", ""),
            "task_description": spec.get("task_description") or "",
            "reference_description": spec.get("reference_description") or "",
            "primary_reference_rationale": "",
            "hypothesis": "",
            "hypothesis_group": "",
            "hypothesis_reference": "",
            "limitations": spec.get("limitations") or list(DEFAULT_LIMITATIONS),
        },
    }
    if spec.get("decision_rule"):
        config["task"]["decision_rule"] = spec["decision_rule"]
    return config


def write_config(run_dir: Path, config: dict) -> Path:
    """Write ``study.yaml`` and validate it with the engine loader."""
    path = run_dir / "study.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    try:
        load_config(path)
    except ConfigError:
        raise
    return path


def validate_spec(spec: dict) -> list[str]:
    """Human-readable problems to show before the config is even written."""
    problems: list[str] = []
    classes = [entry for entry in spec.get("classes", []) if entry.get("name", "").strip()]
    if len(classes) < 2:
        problems.append("Define at least two categories.")
    names = [entry["name"].strip() for entry in classes]
    if len(set(names)) != len(names):
        problems.append("Category names must be unique.")
    import re
    slug = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    for name in names:
        if not slug.match(name):
            problems.append(
                f"Category name {name!r} is not a slug; use letters, digits, dot, "
                "underscore or hyphen (for example 'strongly-positive')."
            )
        definition = next(entry for entry in classes
                          if entry["name"].strip() == name)["definition"].strip()
        if not definition:
            problems.append(f"Category {name!r} needs a definition.")
        elif is_placeholder_definition(definition):
            problems.append(
                f"Category {name!r} still has the placeholder definition; replace "
                "it with what the category means.")
    if not spec.get("dataset_id_column"):
        problems.append("Choose the dataset ID column.")
    if not spec.get("text_column"):
        problems.append("Choose the dataset text column.")
    if not spec.get("benchmark_id_column"):
        problems.append("Choose the benchmark ID column.")
    if not spec.get("label_column"):
        problems.append("Choose the human label column.")
    return problems

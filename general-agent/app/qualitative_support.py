"""Build the qualitative sidecar from interface choices.

The sidecar lives in ``runs/<run_id>/qualitative/settings.yaml`` and points at
``../study.yaml``; the study configuration is never edited. This module contains
no evaluation logic.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app import registry

PILOT_DISAGREEMENTS = 2
PILOT_AGREEMENTS = 1


def settings_path(run_dir: Path) -> Path:
    return Path(run_dir) / "qualitative" / "settings.yaml"


def build_sidecar(provider_entry: dict, model_entry: dict, sample_n: int,
                  control_n: int, seed: int, max_calls: int,
                  max_usd: float | None) -> dict:
    model = registry.model_block(provider_entry, model_entry)
    model["max_attempts"] = 2
    return {
        "study_config": {"path": "../study.yaml"},
        "model": model,
        "sample": {
            "n": int(sample_n),
            "control_n": int(control_n),
            "method": "random",
            "seed": int(seed),
        },
        "pilot": {"disagreements": PILOT_DISAGREEMENTS,
                  "agreements": PILOT_AGREEMENTS},
        "budget": {
            "max_calls": int(max_calls),
            "max_usd": (float(max_usd) if max_usd is not None else None),
            "price_per_million": registry.price_block(model_entry),
        },
        "prompt_version": 1,
        "vocabulary_version": 1,
    }


def write_sidecar(run_dir: Path, sidecar: dict) -> Path:
    path = settings_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def estimate_floor_usd(model_entry: dict, calls: int,
                       chars_per_prompt: int = 1800,
                       output_tokens: int = 300) -> float:
    price = registry.price_block(model_entry)
    per_call = ((chars_per_prompt / 4.0) * float(price["input"])
                + output_tokens * float(price["output"])) / 1_000_000
    return per_call * max(int(calls), 0)


def minimum_calls(sample_n: int, control_n: int) -> int:
    return int(sample_n) + int(control_n)

"""Load and validate stage 08 settings.

The qualitative settings live in a sidecar file so the sealed study
configuration is never edited after classification. A sidecar is optional: with
none, sensible defaults are derived from the study config.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from ..config import ConfigError, load_config

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = PACKAGE_ROOT / "configs" / "qualitative_schema.json"


class QualitativeError(RuntimeError):
    """Raised when stage 08 cannot proceed."""


class QualitativeConfigError(ValueError):
    """Raised when the sidecar settings are missing or invalid."""


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _deep_merge(base: dict, patch: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def default_settings(study_cfg: dict, study_path: Path) -> dict:
    """Defaults derived from the study config, including its model and prices."""
    model = dict(study_cfg["model"])
    model["max_attempts"] = min(int(model.get("max_attempts", 2)), 2)
    sample_n, control_n = 25, 10
    pilot_disagreements, pilot_agreements = 2, 1
    buffer = max(5, math.ceil((sample_n + control_n) * 0.1))
    return {
        "study_config": {"path": str(study_path), "sha256": None},
        "model": model,
        "sample": {
            "n": sample_n,
            "control_n": control_n,
            "method": "random",
            "seed": int(study_cfg["run"]["seed"]),
        },
        "pilot": {"disagreements": pilot_disagreements,
                  "agreements": pilot_agreements},
        "budget": {
            "max_calls": pilot_disagreements + pilot_agreements + sample_n
            + control_n + buffer,
            "max_usd": None,
            "price_per_million": study_cfg["budget"].get("price_per_million"),
        },
        "prompt_version": 1,
        "vocabulary_version": 1,
        "report": dict(study_cfg.get("report", {})),
    }


def load_settings(study_config_path: str | Path,
                  settings_path: str | Path | None = None) -> dict:
    """Return the effective settings, validated and path-resolved."""
    study_path = Path(study_config_path).expanduser().resolve()
    study_cfg = load_config(study_path)
    settings = default_settings(study_cfg, study_path)
    settings_sha = None

    if settings_path:
        path = Path(settings_path).expanduser().resolve()
        if not path.exists():
            raise QualitativeConfigError(f"settings file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise QualitativeConfigError(
                f"{path.name}: top level must be a mapping of sections")
        errors = sorted(Draft202012Validator(load_schema()).iter_errors(raw),
                        key=lambda err: list(err.absolute_path))
        if errors:
            detail = "\n  ".join(
                f"{'/'.join(str(part) for part in err.absolute_path) or '(root)'}: "
                f"{err.message}" for err in errors)
            raise QualitativeConfigError(
                f"{path.name}: schema validation failed\n  {detail}")
        settings = _deep_merge(settings, raw)
        settings_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        declared = raw.get("study_config", {})
        if declared.get("path"):
            declared_path = Path(declared["path"]).expanduser()
            if not declared_path.is_absolute():
                declared_path = (path.parent / declared_path).resolve()
            else:
                declared_path = declared_path.resolve()
            if declared_path != study_path:
                raise QualitativeConfigError(
                    "settings.study_config.path points at "
                    f"{declared_path}, not {study_path}")

    declared_sha = (settings.get("study_config") or {}).get("sha256")
    if declared_sha and declared_sha != study_cfg["_meta"]["config_sha256"]:
        raise QualitativeConfigError(
            "settings.study_config.sha256 does not match the study config; the "
            "study changed after these settings were written")

    _semantic_checks(settings)
    settings["study_config"] = {
        "path": str(study_path),
        "sha256": study_cfg["_meta"]["config_sha256"],
    }
    settings["_meta"] = {
        "settings_path": str(settings_path) if settings_path else None,
        "settings_sha256": settings_sha,
        "study_config_path": str(study_path),
        "study_config_sha256": study_cfg["_meta"]["config_sha256"],
        "run_id": study_cfg["run"]["run_id"],
        "output_dir": str(Path(study_cfg["_meta"]["output_dir"]) / "qualitative"),
        "classes": [
            {"name": entry["name"], "definition": entry["definition"]}
            for entry in study_cfg["labels"]["classes"]
        ],
        "ordered": bool(study_cfg["labels"].get("ordered")),
    }
    return settings


def _semantic_checks(settings: dict) -> None:
    sample = settings["sample"]
    if int(sample["n"]) < 1:
        raise QualitativeConfigError("sample.n must be at least 1")
    if int(sample["control_n"]) < 0:
        raise QualitativeConfigError("sample.control_n cannot be negative")
    pilot = settings["pilot"]
    if int(pilot["disagreements"]) + int(pilot["agreements"]) < 1:
        raise QualitativeConfigError("pilot must include at least one item")
    model = settings["model"]
    if model["provider"] == "opencode_cli" and (
            model.get("base_url") or model.get("api_key_env")):
        raise QualitativeConfigError(
            "model.provider is opencode_cli; base_url/api_key_env belong to "
            "the openai_compatible provider")
    # The pilot items are a subset of the sampled items, so the minimum number
    # of calls is the total number of items to explain; retries consume more.
    budget = settings["budget"]
    minimum = int(sample["n"]) + int(sample["control_n"])
    if int(budget["max_calls"]) < minimum:
        raise QualitativeConfigError(
            f"budget.max_calls is {budget['max_calls']}, below the minimum of "
            f"{minimum} (sample + control)")


def output_dir(settings: dict) -> Path:
    return Path(settings["_meta"]["output_dir"])


def describe_defaults() -> dict:
    return {
        "sample": {"n": 25, "control_n": 10, "method": "random"},
        "pilot": {"disagreements": 2, "agreements": 1},
        "prompt_version": 1,
        "vocabulary_version": 1,
    }

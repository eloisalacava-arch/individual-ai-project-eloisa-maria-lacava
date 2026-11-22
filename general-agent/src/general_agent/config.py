"""Configuration loading, schema validation, and path resolution.

One YAML file per study, validated against ``configs/schema.json``. Every
default applied here is recorded in the run manifest, so a reader can tell an
explicit choice from an inherited one (the D1 lesson in the parent project:
a stale summary line is worse than a verbose one).

Paths inside a config resolve relative to the config file's own directory.
That keeps a study folder movable and makes the shipped examples runnable from
any working directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PACKAGE_ROOT / "configs" / "schema.json"
DEFAULT_OUTPUT_ROOT = "runs"

# Documented defaults. Anything not listed is either required by the schema or
# intentionally left unset so the audit can ask for it.
DEFAULTS = {
    "run": {"output_root": DEFAULT_OUTPUT_ROOT},
    "dataset": {"format": "auto", "delimiter": ",", "encoding": "utf-8"},
    "benchmark": {
        "format": "auto",
        "delimiter": ",",
        "encoding": "utf-8",
        "extra_label_columns": [],
        "reference_labels": {},
        "on_unknown_label": "error",
        "on_id_mismatch": "error",
        "allow_duplicate_ids": False,
    },
    "labels": {"mode": "single_label", "ordered": False},
    "sample": {"n": None, "unique_text": True, "method": "random"},
    "pilot": {"n": 5},
    "task": {"forbid_outside_knowledge": True},
    "model": {"timeout_s": 180, "max_attempts": 3, "concurrency": 1},
    "budget": {"max_usd": None},
    "evaluation": {
        "ci_level": 0.95,
        "bootstrap_draws": 2000,
        "kappa": ["unweighted"],
        "abstention_policy": ["exclude", "count_as_error"],
        "subgroup_columns": [],
        "subgroup_reference": None,
        "subgroup_composites": {},
    },
}


class ConfigError(ValueError):
    """Raised when a config is missing, malformed, or semantically invalid."""


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _format_error(error) -> str:
    location = "/".join(str(part) for part in error.absolute_path) or "(root)"
    return f"{location}: {error.message}"


def _apply_defaults(raw: dict) -> dict:
    cfg = copy.deepcopy(raw)
    for section, values in DEFAULTS.items():
        target = cfg.setdefault(section, {})
        for key, value in values.items():
            target.setdefault(key, copy.deepcopy(value))
    return cfg


def _resolve(base: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _semantic_checks(cfg: dict) -> None:
    names = [entry["name"] for entry in cfg["labels"]["classes"]]
    if len(set(names)) != len(names):
        raise ConfigError(f"labels.classes contains duplicate names: {names}")

    label_map = cfg["benchmark"].get("label_map")
    if label_map:
        empty_keys = [key for key in label_map if not str(key).strip()]
        if empty_keys:
            raise ConfigError("benchmark.label_map has an empty key")
        unknown_targets = sorted(set(label_map.values()) - set(names))
        if unknown_targets:
            raise ConfigError(
                "benchmark.label_map maps to classes that are not declared in "
                f"labels.classes: {unknown_targets} (declared: {names})"
            )

    extra = cfg["benchmark"].get("extra_label_columns", [])
    if cfg["benchmark"].get("label_column") and cfg["benchmark"]["label_column"] in extra:
        raise ConfigError(
            "benchmark.label_column must not also appear in extra_label_columns"
        )

    if cfg["labels"].get("ordered") and "quadratic" not in cfg["evaluation"]["kappa"]:
        # Ordered labels without a weighted kappa silently discards the one
        # statistic that respects the ordering.
        raise ConfigError(
            "labels.ordered is true but evaluation.kappa does not include "
            "'quadratic'; add it or set labels.ordered to false"
        )
    if not cfg["labels"].get("ordered") and "quadratic" in cfg["evaluation"]["kappa"]:
        raise ConfigError(
            "evaluation.kappa requests 'quadratic' but labels.ordered is false; "
            "weighted kappa is only meaningful for an ordinal class set"
        )

    model = cfg["model"]
    if model["provider"] == "opencode_cli":
        if model.get("base_url") or model.get("api_key_env"):
            raise ConfigError(
                "model.provider is opencode_cli; base_url/api_key_env belong to "
                "the openai_compatible provider"
            )
    if model["provider"] == "openai_compatible":
        key_env = model.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            # Not fatal here: the key may be exported only for the classify phase.
            model["_api_key_env_present"] = False
        else:
            model["_api_key_env_present"] = bool(key_env)

    if cfg.get("dataset", {}).get("predictions"):
        predictions = cfg["dataset"]["predictions"]
        predictions.setdefault("reuse", True)
        predictions.setdefault("format", cfg["dataset"]["format"])
        predictions.setdefault("delimiter", cfg["dataset"]["delimiter"])
        predictions.setdefault("encoding", cfg["dataset"]["encoding"])


def load_config(path: str | Path) -> dict:
    """Load, validate, resolve paths, and attach metadata."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping of sections")

    schema = load_schema()
    errors = sorted(Draft202012Validator(schema).iter_errors(raw),
                    key=lambda err: list(err.absolute_path))
    if errors:
        detail = "\n  ".join(_format_error(err) for err in errors)
        raise ConfigError(f"{path.name}: schema validation failed\n  {detail}")

    cfg = _apply_defaults(raw)
    base = path.parent
    cfg["dataset"]["path"] = _resolve(base, cfg["dataset"]["path"])
    cfg["benchmark"]["path"] = _resolve(base, cfg["benchmark"]["path"])
    if cfg["dataset"].get("predictions"):
        cfg["dataset"]["predictions"]["path"] = _resolve(
            base, cfg["dataset"]["predictions"]["path"]
        )
    cfg["run"]["output_root"] = _resolve(base, cfg["run"]["output_root"])

    _semantic_checks(cfg)
    cfg["_meta"] = {
        "config_path": str(path),
        "config_dir": str(base),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_dir": str(Path(cfg["run"]["output_root"]) / cfg["run"]["run_id"]),
    }
    return cfg


def class_names(cfg: dict) -> list[str]:
    return [entry["name"] for entry in cfg["labels"]["classes"]]


def label_columns(cfg: dict) -> list[str]:
    """Primary reference first, then extras, skipping unset (null) entries."""
    columns = [cfg["benchmark"]["label_column"]]
    columns.extend(cfg["benchmark"].get("extra_label_columns", []))
    return [column for column in columns if column]


def describe_defaults() -> dict:
    """Defaults applied by :func:`load_config`, for the manifest and the README."""
    return copy.deepcopy(DEFAULTS)


def applied_defaults(config_path: str | Path) -> dict:
    """Only the defaults actually used, i.e. keys absent from the raw config.

    Printing every default would present an explicit choice as if it were
    inherited - the stale-summary failure mode this project is designed to avoid.
    """
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    applied: dict[str, dict] = {}
    for section, values in DEFAULTS.items():
        section_raw = raw.get(section) or {}
        missing = {key: value for key, value in values.items() if key not in section_raw}
        if missing:
            applied[section] = missing
    return applied
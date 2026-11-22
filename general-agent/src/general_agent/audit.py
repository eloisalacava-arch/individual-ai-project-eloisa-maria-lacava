"""Stage 01 - dataset audit and column discovery.

Reads the configured dataset and benchmark, proposes or confirms the column
mapping, and records every automatic choice. Produces, under
``runs/<run_id>/``:

  decisions.json   resolved mapping, each decision with its reason, and any
                   questions that block the run
  audit.txt        full audit trail: hashes, column profiles, row accounting,
                   PASS/FAIL checks, distributions, warnings
  manifest.json    config hash, input hashes, seed, model identity, checks

This stage never calls a model and never writes outside ``runs/<run_id>/``.
The benchmark's labels are loaded here because auditing them is the job of this
stage; from stage 02 onward only a sealed, separate file holds them.

Row-count rule (AGENTS.md): every filter, join, or exclusion prints the row
count before and after, with the reason.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .config import applied_defaults, class_names, describe_defaults, label_columns, load_config
from .discover import (
    LABEL_NAME_HINTS,
    map_labels,
    observation_text,
    profile_frame,
    propose,
    redact_benchmark_examples,
)
from .io import (
    file_fingerprint,
    normalize_ids,
    read_table,
    write_json_atomic,
    write_text_atomic,
)
from .provenance import load_manifest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


class AuditLog:
    """Collects the audit trail, PASS/FAIL checks, and warnings."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.checks: list[dict] = []
        self.warnings: list[str] = []
        self.row_accounting: list[dict] = []

    def log(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def heading(self, title: str) -> None:
        self.log("")
        self.log(f"== {title} ==")

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        self.log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.log(f"[WARN] {message}")

    def filter_rows(self, name: str, before: int, after: int, reason: str) -> None:
        dropped = before - after
        self.row_accounting.append({
            "step": name,
            "rows_before": before,
            "rows_after": after,
            "rows_dropped": dropped,
            "reason": reason,
        })
        self.log(f"rows: {before} -> {after} ({dropped} dropped; reason: {reason})")

    @property
    def failed(self) -> list[str]:
        return [entry["name"] for entry in self.checks if not entry["ok"]]


def _text_summary(text: pd.Series) -> dict:
    lengths = text.astype("string").str.len().dropna()
    return {
        "n": int(len(text)),
        "n_blank": int(text.astype("string").str.strip().eq("").fillna(False).sum()),
        "median_chars": round(float(lengths.median()), 1) if len(lengths) else 0.0,
        "min_chars": int(lengths.min()) if len(lengths) else 0,
        "max_chars": int(lengths.max()) if len(lengths) else 0,
    }


def _print_profiles(log: AuditLog, role: str, profiles: list[dict]) -> None:
    log.log(f"{role}: {len(profiles)} columns")
    for profile in profiles:
        log.log(
            f"  - {profile['name']:<24} {profile['dtype']:<10} "
            f"unique={profile['n_unique']:<5} median_chars={profile['median_chars']:<8} "
            f"blank={profile['n_blank_or_null']}"
        )


def _leakage_checks(log: AuditLog, dataset: pd.DataFrame, benchmark: pd.DataFrame,
                    mapping: dict, references: list[str]) -> None:
    """Fail if the dataset carries the benchmark label, by name or by value."""
    colliding = [column for column in references if column in dataset.columns]
    log.check(
        "dataset_excludes_reference_label_columns",
        not colliding,
        f"reference label columns found in the dataset: {colliding}"
        if colliding else f"none of {references} appear in the dataset columns",
    )

    dataset_ids = normalize_ids(dataset[mapping["id_column"]]).dropna()
    benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]]).dropna()
    common = sorted(set(dataset_ids) & set(benchmark_ids))
    if not common:
        log.warn("no matched IDs: value-based leakage check skipped")
        return

    left = dataset.set_index(dataset_ids).loc[common]
    right = benchmark.set_index(benchmark_ids).loc[common]
    for column in dataset.columns:
        if column == mapping["id_column"]:
            continue
        left_values = left[column].astype("string")
        for reference in references:
            right_values = right[reference].astype("string")
            comparable = left_values.notna() & right_values.notna()
            if comparable.sum() == 0:
                continue
            match_rate = float((left_values == right_values)[comparable].mean())
            if match_rate >= 0.98 and left_values[comparable].nunique() > 1:
                log.check(
                    f"dataset_column_not_a_label_copy[{column}]",
                    False,
                    f"dataset column '{column}' equals benchmark reference '{reference}' "
                    f"on {match_rate:.1%} of matched rows; this would leak labels into the prompt",
                )
            elif match_rate >= 0.9 and LABEL_NAME_HINTS.search(column.lower()):
                log.warn(
                    f"dataset column '{column}' matches reference '{reference}' on "
                    f"{match_rate:.1%} of matched rows; check that it is not a label copy"
                )


def _predictions_checks(log: AuditLog, cfg: dict, dataset: pd.DataFrame,
                        mapping: dict, classes: list[str]) -> dict | None:
    predictions_cfg = cfg["dataset"].get("predictions")
    if not predictions_cfg:
        return None

    log.heading("Reusable predictions")
    frame = read_table(
        predictions_cfg["path"],
        predictions_cfg["format"],
        predictions_cfg["delimiter"],
        predictions_cfg["encoding"],
    )
    log.log(f"rows read from {Path(predictions_cfg['path']).name}: {len(frame)} "
            "(no filter, no drop at load)")
    log.log(f"columns: {list(frame.columns)}")

    id_column = predictions_cfg.get("id_column") or mapping.get("id_column")
    label_column = predictions_cfg["label_column"]
    status_column = predictions_cfg.get("status_column")
    model_column = predictions_cfg.get("model_column")

    log.check("predictions_nonzero_rows", len(frame) > 0, f"{len(frame)} rows")
    log.check("predictions_columns_present",
              id_column in frame.columns and label_column in frame.columns,
              f"id='{id_column}' label='{label_column}' in {list(frame.columns)}")
    if id_column not in frame.columns or label_column not in frame.columns:
        return {"rows": len(frame), "usable": False}

    ids = normalize_ids(frame[id_column])
    duplicated = int(ids.duplicated(keep="first").sum())
    log.check("predictions_ids_unique", duplicated == 0,
              f"{duplicated} duplicate ID(s) of {len(frame)} rows")

    dataset_ids = set(normalize_ids(dataset[mapping["id_column"]]).dropna())
    prediction_ids = set(ids.dropna())
    missing = sorted(dataset_ids - prediction_ids)
    extra = sorted(prediction_ids - dataset_ids)
    log.check("predictions_cover_dataset_ids", not missing and not extra,
              f"{len(missing)} dataset ID(s) without a prediction, "
              f"{len(extra)} prediction ID(s) not in the dataset "
              f"(examples: missing {missing[:3]}, extra {extra[:3]})")

    labels = frame[label_column].astype("string").str.strip()
    invalid = sorted(set(labels.dropna()) - set(classes))
    n_missing = int(labels.isna().sum())
    log.check("predictions_labels_in_declared_classes", not invalid,
              f"undeclared values: {invalid}" if invalid else f"all values in {classes}")
    if n_missing:
        log.warn(f"{n_missing} row(s) have no prediction label; they must be reported "
                 "as failures or excluded with a stated policy, never imputed")
    log.check("predictions_no_missing_labels", n_missing == 0,
              f"{n_missing} missing of {len(frame)}")

    if status_column and status_column in frame.columns:
        statuses = sorted(set(frame[status_column].astype("string").dropna()))
        log.log(f"status values in '{status_column}': {statuses}")
        non_ok = int((frame[status_column].astype("string") != "ok").sum())
        log.check("predictions_all_ok", non_ok == 0,
                  f"{non_ok} row(s) have status != 'ok'")
    elif status_column:
        log.check("predictions_status_column_present", False,
                  f"status column '{status_column}' not in {list(frame.columns)}")

    if model_column and model_column in frame.columns:
        models = sorted(set(frame[model_column].astype("string").dropna()))
        log.check("predictions_single_model", len(models) == 1,
                  f"models seen: {models}")

    log.log(f"reuse mode: {predictions_cfg.get('reuse')} (true means zero model calls "
            "for this stage)")
    return {
        "rows": len(frame),
        "usable": not invalid and not missing and not extra and duplicated == 0,
        "n_missing_labels": n_missing,
        "models": sorted(set(frame[model_column].astype("string").dropna()))
        if model_column and model_column in frame.columns else [],
    }


def run_audit(config_path: str | Path, strict: bool = False) -> int:
    cfg = load_config(config_path)
    out_dir = Path(cfg["_meta"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log = AuditLog()
    log.log(f"script: general-agent audit (version {__version__})")
    log.log(f"started: {_now()}")
    log.log(f"config: {cfg['_meta']['config_path']}")
    log.log(f"config sha256: {cfg['_meta']['config_sha256']}")
    log.log(f"run_id: {cfg['run']['run_id']}   seed: {cfg['run']['seed']}")
    log.log(f"provider: {cfg['model']['provider']}   model: {cfg['model']['name']}")
    temperature = cfg["model"].get("temperature")
    log.log(f"temperature: {temperature!r} (requested only; support is not guaranteed - "
            "see SPEC.md section 2)")

    classes = class_names(cfg)
    references = label_columns(cfg)
    log.log(f"declared classes ({len(classes)}): {classes}")
    log.log(f"benchmark references configured: primary="
            f"{cfg['benchmark']['label_column']!r}, "
            f"extras={cfg['benchmark'].get('extra_label_columns', [])}")

    # ---- inputs ----------------------------------------------------------
    log.heading("Inputs")
    try:
        dataset = read_table(cfg["dataset"]["path"], cfg["dataset"]["format"],
                             cfg["dataset"]["delimiter"], cfg["dataset"]["encoding"])
        benchmark = read_table(cfg["benchmark"]["path"], cfg["benchmark"]["format"],
                               cfg["benchmark"]["delimiter"], cfg["benchmark"]["encoding"])
    except (FileNotFoundError, ValueError) as exc:
        log.log(f"ABORT: {exc}")
        write_text_atomic(out_dir / "audit.txt", "\n".join(log.lines) + "\n")
        return 1

    log.log(f"dataset rows: {len(dataset)}  columns: {list(dataset.columns)}")
    log.log(f"benchmark rows: {len(benchmark)}  columns: {list(benchmark.columns)}")
    for name, path in (("dataset", cfg["dataset"]["path"]),
                       ("benchmark", cfg["benchmark"]["path"])):
        fingerprint = file_fingerprint(path)
        log.log(f"{name} sha256: {fingerprint.get('sha256', 'MISSING')}")

    _print_profiles(log, "dataset", profile_frame(dataset, "dataset"))
    _print_profiles(log, "benchmark", profile_frame(benchmark, "benchmark"))

    # ---- discovery -------------------------------------------------------
    log.heading("Column discovery")
    proposal = propose(dataset, benchmark, cfg)
    for decision in proposal.decisions:
        log.log(f"[decision] {decision}")

    decisions_path = out_dir / "decisions.json"
    decisions_payload = {
        "run_id": cfg["run"]["run_id"],
        "created_at": _now(),
        "resolved_mapping": proposal.mapping,
        "decisions": proposal.decisions,
        "questions": proposal.questions,
        "stats": proposal.stats,
        "profiles": {
            "dataset": proposal.profiles["dataset"],
            "benchmark": redact_benchmark_examples(proposal.profiles["benchmark"]),
        },
        "declared_classes": classes,
        "references": references,
        "label_map": cfg["benchmark"].get("label_map"),
        "status": "needs_clarification" if proposal.questions else "resolved",
    }
    write_json_atomic(decisions_path, decisions_payload)
    log.log(f"decisions written: {decisions_path.name}")

    manifest: dict = {
        "run_id": cfg["run"]["run_id"],
        "phase": "01_audit",
        "tool": {"name": "general-agent", "version": __version__},
        "created_at": _now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "config": {"path": cfg["_meta"]["config_path"],
                   "sha256": cfg["_meta"]["config_sha256"]},
        "seed": cfg["run"]["seed"],
        "class_names": classes,
        "references": references,
        "model": {
            "provider": cfg["model"]["provider"],
            "name": cfg["model"]["name"],
            "temperature_requested": cfg["model"].get("temperature"),
            "temperature_status": "not_requested"
            if cfg["model"].get("temperature") is None
            else "requested_unverified",
            "temperature_note": cfg["model"].get("temperature_note", ""),
            "max_output_tokens": cfg["model"].get("max_output_tokens"),
            "timeout_s": cfg["model"].get("timeout_s"),
            "max_attempts": cfg["model"].get("max_attempts"),
            "concurrency": cfg["model"].get("concurrency"),
            "api_key_env": cfg["model"].get("api_key_env"),
            "api_key_env_present": cfg["model"].get("_api_key_env_present"),
        },
        "budget": cfg["budget"],
        "defaults_applied": applied_defaults(cfg["_meta"]["config_path"]),
        "documented_defaults": describe_defaults(),
        "inputs": {
            "dataset": file_fingerprint(cfg["dataset"]["path"]),
            "benchmark": file_fingerprint(cfg["benchmark"]["path"]),
            "predictions": file_fingerprint(cfg["dataset"]["predictions"]["path"])
            if cfg["dataset"].get("predictions") else None,
        },
        "row_counts": {"dataset": len(dataset), "benchmark": len(benchmark)},
    }

    if proposal.questions:
        log.heading("Clarification needed")
        for question in proposal.questions:
            log.log(f"[?] {question['field']}: {question['reason']}")
            if question["options"]:
                log.log(f"    options: {json.dumps(question['options'], ensure_ascii=False)}")
            log.log(f"    resolve by: {question['how_to_resolve']}")
        log.log("")
        log.log(f"result: CLARIFICATION NEEDED ({len(proposal.questions)} question(s)); "
                "no dataset rows were dropped")
        manifest["checks"] = {"passed": 0, "total": 0, "failed": [],
                              "status": "needs_clarification"}
        manifest["warnings"] = log.warnings
        manifest["stages"] = load_manifest(out_dir).get("stages", {})
        write_json_atomic(out_dir / "manifest.json", manifest)
        write_text_atomic(out_dir / "audit.txt", "\n".join(log.lines) + "\n")
        return 2

    mapping = proposal.mapping
    references = [mapping["label_column"]] + list(
        mapping.get("extra_label_columns", [])
    )
    log.log(f"resolved: id='{mapping['id_column']}', "
            f"benchmark id='{mapping['id_column_benchmark']}', "
            f"text={mapping.get('text_column') or 'template:' + mapping.get('text_template', '')}, "
            f"label='{mapping['label_column']}'")
    manifest["references"] = references

    # ---- integrity checks ------------------------------------------------
    log.heading("Integrity checks")
    dataset_ids = normalize_ids(dataset[mapping["id_column"]])
    benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]])

    log.check("dataset_nonzero_rows", len(dataset) > 0, f"{len(dataset)} rows")
    log.check("benchmark_nonzero_rows", len(benchmark) > 0, f"{len(benchmark)} rows")
    log.check("dataset_ids_complete", int(dataset_ids.isna().sum()) == 0,
              f"{int(dataset_ids.isna().sum())} blank/null ID(s) of {len(dataset)}")
    log.check("benchmark_ids_complete", int(benchmark_ids.isna().sum()) == 0,
              f"{int(benchmark_ids.isna().sum())} blank/null ID(s) of {len(benchmark)}")
    log.check("dataset_ids_unique", int(dataset_ids.duplicated().sum()) == 0,
              f"{int(dataset_ids.duplicated().sum())} duplicate ID(s) of {len(dataset)}")
    log.check("benchmark_ids_unique", int(benchmark_ids.duplicated().sum()) == 0,
              f"{int(benchmark_ids.duplicated().sum())} duplicate ID(s) of {len(benchmark)}")

    # ---- ID alignment / inner join --------------------------------------
    log.heading("Row alignment")
    before_dataset, before_benchmark = len(dataset), len(benchmark)
    only_dataset = sorted(set(dataset_ids.dropna()) - set(benchmark_ids.dropna()))
    only_benchmark = sorted(set(benchmark_ids.dropna()) - set(dataset_ids.dropna()))
    alignment = {
        "n_dataset_ids": int(len(set(dataset_ids.dropna()))),
        "n_benchmark_ids": int(len(set(benchmark_ids.dropna()))),
        "n_only_dataset": len(only_dataset),
        "n_only_benchmark": len(only_benchmark),
        "matched": int(len(set(dataset_ids.dropna()) & set(benchmark_ids.dropna()))),
    }
    log.log(f"ID sets: dataset {alignment['n_dataset_ids']}, "
            f"benchmark {alignment['n_benchmark_ids']}, "
            f"matched {alignment['matched']}, "
            f"dataset-only {len(only_dataset)}, benchmark-only {len(only_benchmark)}")

    if only_dataset or only_benchmark:
        if cfg["benchmark"]["on_id_mismatch"] == "inner_join":
            common = set(dataset_ids.dropna()) & set(benchmark_ids.dropna())
            dataset = dataset[dataset_ids.isin(common)].reset_index(drop=True)
            log.filter_rows("dataset restricted to matched IDs", before_dataset,
                            len(dataset),
                            f"{len(only_dataset)} ID(s) with no benchmark row")
            benchmark = benchmark[benchmark_ids.isin(common)].reset_index(drop=True)
            log.filter_rows("benchmark restricted to matched IDs", before_benchmark,
                            len(benchmark),
                            f"{len(only_benchmark)} ID(s) with no dataset row")
            dataset_ids = normalize_ids(dataset[mapping["id_column"]])
            benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]])
        else:
            log.check("id_sets_match", False,
                      f"{len(only_dataset)} dataset-only and {len(only_benchmark)} "
                      "benchmark-only ID(s); on_id_mismatch is 'error'")
    log.check("id_sets_match_after_policy",
              set(dataset_ids.dropna()) == set(benchmark_ids.dropna()),
              f"{len(dataset)} dataset rows, {len(benchmark)} benchmark rows, "
              f"{alignment['matched']} shared IDs")

    # ---- observation text ------------------------------------------------
    log.heading("Observation text")
    text = observation_text(dataset, mapping)
    text_summary = _text_summary(text.astype("string"))
    log.log(f"observation text: n={text_summary['n']}, blank={text_summary['n_blank']}, "
            f"chars median={text_summary['median_chars']} "
            f"min={text_summary['min_chars']} max={text_summary['max_chars']}")
    log.check("dataset_text_nonblank", text_summary["n_blank"] == 0,
              f"{text_summary['n_blank']} blank observation text(s)")
    if text_summary["median_chars"] < 20 and len(text):
        log.warn(f"median observation length is {text_summary['median_chars']} characters; "
                 "the text column may be an ID or a short field")
    duplicate_text = int(text.duplicated(keep="first").sum())
    log.log(f"duplicate observation texts: {duplicate_text} (information only; the report "
            "must state whether duplicates count as independent observations)")

    # ---- label domains ---------------------------------------------------
    log.heading("Label domains")
    mapped: dict[str, pd.Series] = {}
    for column in references:
        if column not in benchmark.columns:
            log.check(f"label_column_present[{column}]", False,
                      f"'{column}' not in benchmark columns {list(benchmark.columns)}")
            continue
        values, unmapped = map_labels(
            benchmark[column], cfg["benchmark"].get("label_map"), classes
        )
        mapped[column] = values
        n_unmapped = int(values.isna().sum())
        policy = cfg["benchmark"]["on_unknown_label"]
        ok = n_unmapped == 0 or policy == "exclude"
        log.check(f"label_domain[{column}]", ok,
                  f"{n_unmapped} row(s) missing or outside {classes} "
                  f"after label_map; policy={policy}"
                  + (f"; unmapped raw values: {unmapped}" if unmapped else ""))

    primary = mapping["label_column"]
    if primary in mapped:
        primary_values = mapped[primary]
        before = len(benchmark)
        keep = primary_values.notna().to_numpy()
        if not keep.all():
            benchmark = benchmark[keep].reset_index(drop=True)
            for column in mapped:
                mapped[column] = mapped[column][keep].reset_index(drop=True)
            log.filter_rows("rows without a usable primary label", before, len(benchmark),
                            f"{int((~keep).sum())} row(s) missing or unclassifiable under "
                            f"policy '{cfg['benchmark']['on_unknown_label']}'")
            dataset_ids = normalize_ids(dataset[mapping["id_column"]])
            benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]])

        present = sorted(set(mapped[primary].dropna()))
        missing_classes = [name for name in classes if name not in present]
        log.check("all_declared_classes_present_in_primary", not missing_classes,
                  f"absent from the primary reference: {missing_classes}; "
                  f"present: {present}")

        distribution = mapped[primary].value_counts().to_dict()
        log.log(f"primary reference distribution ({primary}): "
                f"{ {str(k): int(v) for k, v in distribution.items()} }")

        for column in references:
            if column == primary or column not in mapped:
                continue
            available = int(mapped[column].notna().sum())
            excluded = len(mapped[column]) - available
            log.log(f"reference '{column}': {available} of {len(mapped[column])} rows "
                    f"analysable ({excluded} excluded: missing or unclassifiable)"

                    + (f"; distribution: "
                       f"{ {str(k): int(v) for k, v in mapped[column].value_counts().to_dict().items()} }"
                       if available else ""))

    # ---- leakage ---------------------------------------------------------
    log.heading("Leakage checks")
    _leakage_checks(log, dataset, benchmark, mapping, references)

    # ---- reusable predictions -------------------------------------------
    predictions_summary = _predictions_checks(log, cfg, dataset, mapping, classes)

    # ---- finish ----------------------------------------------------------
    failed = log.failed
    strict_failure = strict and bool(log.warnings)
    status = "PASS" if not failed and not strict_failure else "FAIL"

    log.heading("Result")
    log.log(f"checks passed: {len(log.checks) - len(failed)}/{len(log.checks)}")
    log.log(f"warnings: {len(log.warnings)}")
    log.log(f"result: {status}"
            + (" (strict mode: warnings count as failures)" if strict else ""))
    if failed:
        log.log(f"failed checks: {failed}")

    manifest.update({
        "row_counts": {
            "dataset_input": before_dataset,
            "benchmark_input": before_benchmark,
            "dataset_analysed": len(dataset),
            "benchmark_analysed": len(benchmark),
            "matched_ids": alignment["matched"],
            "duplicate_text": duplicate_text,
        },
        "text_summary": text_summary,
        "label_distributions": {
            str(column): {str(k): int(v) for k, v in values.value_counts().to_dict().items()}
            for column, values in mapped.items()
        },
        "predictions": predictions_summary,
        "row_accounting": log.row_accounting,
        "checks": {
            "passed": len(log.checks) - len(failed),
            "total": len(log.checks),
            "failed": failed,
            "status": status,
        },
        "warnings": log.warnings,
        "decisions_sha256": file_fingerprint(decisions_path)["sha256"],
    })
    previous_stages = load_manifest(out_dir).get("stages", {})
    manifest["stages"] = {
        **previous_stages,
        "01_audit": {
            "finished_at": _now(),
            "checks": manifest["checks"],
            "row_counts": manifest["row_counts"],
            "warnings": log.warnings,
            "row_accounting": log.row_accounting,
            "decisions_sha256": manifest["decisions_sha256"],
        },
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    write_text_atomic(out_dir / "audit.txt", "\n".join(log.lines) + "\n")
    log.log(f"audit written: {out_dir / 'audit.txt'}")
    log.log(f"manifest written: {out_dir / 'manifest.json'}")
    log.log(f"finished: {_now()}")

    return 0 if status == "PASS" else 1
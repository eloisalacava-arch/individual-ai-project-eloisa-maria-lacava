"""Stage 05 - evaluate frozen predictions against the sealed reference.

Order matters and is logged: freeze first, open the benchmark second. Every
number is written to a CSV under ``runs/<run_id>/results/``; this function never
prints a metric that is not also in a file.

Nothing here is causal. A disagreement with a majority-vote reference is not a
confirmed model error, and the tables say so explicitly rather than blurring the
two.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..config import class_names, load_config
from ..provenance import file_sha256, now_iso, update_manifest
from . import intervals, metrics
from .freeze import FreezeError, freeze_predictions
from .import_reuse import ReuseError, materialise_predictions
from .join import JoinError, join_frame, load_benchmark

RESULTS_NAME = "results"


def _save(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, float_format="%.6f")
    return path


def _wilson_row(k: int, n: int, level: float) -> tuple[float, float, float]:
    return (k / n if n else math.nan, *intervals.wilson(k, n, level))


def run_evaluation(config_path: str | Path) -> int:
    cfg = load_config(config_path)
    out_dir = Path(cfg["_meta"]["output_dir"])
    results_dir = out_dir / RESULTS_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    classes = class_names(cfg)
    level = float(cfg["evaluation"]["ci_level"])
    draws = int(cfg["evaluation"]["bootstrap_draws"])
    seed = int(cfg["run"]["seed"])
    weighted_requested = "quadratic" in cfg["evaluation"]["kappa"]

    lines: list[str] = []
    checks: list[dict] = []

    def log(message: str = "") -> None:
        print(message)
        lines.append(message)

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    log(f"script: general-agent evaluate (version {__version__})")
    log(f"started: {now_iso()}")
    log(f"config: {cfg['_meta']['config_path']}")
    log(f"run_id: {cfg['run']['run_id']} | seed: {seed} | CI level: {level} | "
        f"bootstrap draws: {draws}")
    log(f"declared classes (order used for ordinal distance): {classes}")

    # ---- 1. freeze BEFORE touching the benchmark -------------------------
    log("")
    log("== Step 1: freeze predictions (benchmark not yet opened) ==")
    if not (out_dir / "predictions.csv").exists():
        try:
            imported = materialise_predictions(out_dir, cfg)
        except ReuseError as exc:
            log(f"ABORT: {exc}")
            (results_dir / "evaluation_log.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            return 1
        if imported:
            log(f"reused predictions imported: {imported['n_rows']} rows from "
                f"{imported['source_path']} (sha256 {imported['source_sha256'][:16]}...); "
                "0 model calls")
        else:
            log("no predictions.csv and no reuse source configured")
    try:
        freeze = freeze_predictions(out_dir, results_dir, classes)
    except FreezeError as exc:
        log(f"ABORT: {exc}")
        (results_dir / "evaluation_log.txt").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
        return 1
    log(f"frozen: {freeze['n_rows']} rows, sha256 {freeze['frozen_sha256'][:16]}...")
    log(f"status counts: {freeze['status_counts']}")
    log(f"models: {freeze['models']}")
    check("frozen_file_matches_source",
          freeze["frozen_sha256"] == freeze["source_sha256"],
          "the scored copy is byte-identical to predictions.csv at freeze time")

    frozen = pd.read_csv(results_dir / "frozen_predictions.csv", dtype="string",
                         keep_default_na=False)

    # ---- 2. open the benchmark and join ---------------------------------
    log("")
    log("== Step 2: open sealed reference and join ==")
    benchmark_read_at = now_iso()
    sealed_path = out_dir / "sealed" / "benchmark.csv"
    try:
        benchmark = load_benchmark(out_dir)
    except (FileNotFoundError, ValueError) as exc:
        log(f"ABORT: {exc}")
        (results_dir / "evaluation_log.txt").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
        return 1
    log(f"sealed reference read at {benchmark_read_at}: {len(benchmark)} rows, "
        f"sha256 {file_sha256(sealed_path)[:16]}...")
    check("benchmark_opened_after_freeze",
          benchmark_read_at >= freeze["frozen_at"],
          f"freeze {freeze['frozen_at']} <= read {benchmark_read_at}")

    try:
        joined, stats, join_checks = join_frame(frozen, benchmark, cfg, log)
    except JoinError as exc:
        log(f"ABORT: {exc}")
        (results_dir / "evaluation_log.txt").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
        return 1
    checks.extend(join_checks)

    if len(joined) == 0:
        log("ABORT: no rows survived the join")
        (results_dir / "evaluation_log.txt").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
        return 1

    y_true = metrics.label_indices(joined["reference"], classes)
    y_pred = metrics.label_indices(joined["predicted"], classes)
    n = len(joined)
    k = len(classes)
    matrix = metrics.confusion_matrix(y_true, y_pred, k)
    per_class = metrics.per_class(matrix)
    core = metrics.all_core_metrics(matrix)
    boot_core = intervals.bootstrap_core(y_true, y_pred, k, draws, seed, level)
    boot_class = intervals.bootstrap_per_class(y_true, y_pred, k, draws, seed, level)
    ordinal = metrics.ordinal_summary(y_true, y_pred, k)

    # ---- 3. tables -------------------------------------------------------
    log("")
    log("== Step 3: metrics ==")
    n_correct = int(np.trace(matrix))
    accuracy, acc_low, acc_high = _wilson_row(n_correct, n, level)

    _save(pd.DataFrame([
        {"step": "predictions (frozen copy)", "n": stats["predictions_rows"],
         "reason": "frozen before benchmark access"},
        {"step": "benchmark (sealed reference)", "n": stats["benchmark_rows"],
         "reason": "opened after freeze"},
        {"step": "benchmark rows without a prediction",
         "n": stats["only_in_benchmark"],
         "reason": "reference row has no prediction; excluded from the join"},
        {"step": "matched on id", "n": stats["matched_rows"],
         "reason": "inner join, one-to-one"},
        {"step": "dropped predictions", "n": stats["dropped_predictions"],
         "reason": "no matching reference row"},
        {"step": "excluded failed predictions", "n": stats["excluded_not_ok"],
         "reason": "status != 'ok'; never imputed"},
        {"step": "analysed", "n": stats["analysed_rows"],
         "reason": "rows entering the metrics"},
    ]), results_dir / "t01_join_and_sample.csv")

    _save(pd.DataFrame([{
        "n": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "accuracy_ci_method": "wilson",
        "balanced_accuracy": core["balanced_accuracy"],
        "ba_ci_low": boot_core["balanced_accuracy"]["low"],
        "ba_ci_high": boot_core["balanced_accuracy"]["high"],
        "ba_ci_method": "bootstrap_percentile",
        "bootstrap_draws": draws,
        "bootstrap_valid_draws_ba": boot_core["balanced_accuracy"]["valid"],
        "seed": seed,
        "ci_level": level,
    }]), results_dir / "t02_accuracy.csv")

    kappa_rows = [{
        "metric": "cohen_kappa_unweighted",
        "weighting": "none",
        "value": core["kappa_unweighted"],
        "ci_low": boot_core["kappa_unweighted"]["low"],
        "ci_high": boot_core["kappa_unweighted"]["high"],
        "ci_method": "bootstrap_percentile",
        "bootstrap_draws": draws,
        "valid_draws": boot_core["kappa_unweighted"]["valid"],
        "seed": seed,
        "note": "chance-corrected agreement; depends on the marginal distribution",
    }]
    if weighted_requested:
        kappa_rows.append({
            "metric": "cohen_kappa_quadratic_weighted",
            "weighting": "quadratic",
            "value": core["kappa_quadratic"],
            "ci_low": boot_core["kappa_quadratic"]["low"],
            "ci_high": boot_core["kappa_quadratic"]["high"],
            "ci_method": "bootstrap_percentile",
            "bootstrap_draws": draws,
            "valid_draws": boot_core["kappa_quadratic"]["valid"],
            "seed": seed,
            "note": "respects the ordinal order negative < neutral < positive",
        })
    _save(pd.DataFrame(kappa_rows), results_dir / "t03_kappa.csv")

    # Confusion matrix, orientation stated in the header.
    cm = pd.DataFrame(matrix, index=classes, columns=classes)
    cm.index.name = "human\\predicted"
    cm.loc["total_predicted"] = cm.sum(axis=0)
    cm["total_human"] = cm.sum(axis=1)
    cm.to_csv(results_dir / "t04_confusion_matrix.csv", float_format="%.6f")

    per_rows = []
    for index, name in enumerate(classes):
        entry = per_class[index]
        recall = entry["recall"]
        recall_low, recall_high = intervals.wilson(entry["tp"], entry["support"], level)
        per_rows.append({
            "class": name,
            "n_reference": entry["support"],
            "n_predicted": entry["predicted"],
            "tp": entry["tp"],
            "fp": entry["fp"],
            "fn": entry["fn"],
            "precision": entry["precision"],
            "precision_ci_low": boot_class[index]["precision_low"],
            "precision_ci_high": boot_class[index]["precision_high"],
            "precision_ci_method": "bootstrap_percentile",
            "recall": recall,
            "recall_ci_low": recall_low,
            "recall_ci_high": recall_high,
            "recall_ci_method": "wilson",
            "f1": entry["f1"],
            "f1_ci_low": boot_class[index]["f1_low"],
            "f1_ci_high": boot_class[index]["f1_high"],
            "f1_ci_method": "bootstrap_percentile",
            "bootstrap_valid_draws": boot_class[index]["precision_valid"],
            "ci_level": level,
        })
    _save(pd.DataFrame(per_rows), results_dir / "t05_per_class_metrics.csv")

    human_counts = np.bincount(y_true, minlength=k)
    ai_counts = np.bincount(y_pred, minlength=k)
    _save(pd.DataFrame([{
        "class": name,
        "human_n": int(human_counts[index]),
        "human_share": human_counts[index] / n,
        "ai_n": int(ai_counts[index]),
        "ai_share": ai_counts[index] / n,
        "difference_ai_minus_human": (ai_counts[index] - human_counts[index]) / n,
    } for index, name in enumerate(classes)]),
        results_dir / "t06_class_distribution.csv")

    ordinal_rows = []
    for distance in range(k):
        count = ordinal["distances"].get(distance, 0)
        share = count / n
        low, high = intervals.wilson(count, n, level)
        # Check the maximum distance before adjacency: for k == 2 they are the
        # same value, and the maximum is the correct description.
        if distance == 0:
            bucket = "agree"
        elif distance == k - 1:
            bucket = "opposite"
        elif distance == 1:
            bucket = "adjacent"
        else:
            bucket = f"distance_{distance}"
        ordinal_rows.append({
            "ordinal_distance": distance,
            "bucket": bucket,
            "n": count,
            "share": share,
            "ci_low": low,
            "ci_high": high,
            "ci_method": "wilson",
        })
    _save(pd.DataFrame(ordinal_rows), results_dir / "t07_ordinal_errors.csv")

    # ---- 4. disagreements vs errors -------------------------------------
    log("")
    log("== Step 4: disagreements (not confirmed errors) ==")
    mask = y_true != y_pred
    disagreements = joined.loc[mask, ["id", "reference", "predicted"]].copy()
    distances = np.abs(y_true[mask] - y_pred[mask])
    disagreements["ordinal_distance"] = distances
    disagreements["severity"] = np.where(distances == 1, "adjacent", "opposite")
    disagreements["direction"] = (
        disagreements["reference"] + "->" + disagreements["predicted"])
    disagreements["involves_neutral"] = (
        (disagreements["reference"] == "neutral")
        | (disagreements["predicted"] == "neutral")
    )
    disagreements["error_status"] = "unconfirmed_disagreement"
    primary_column = cfg["benchmark"]["label_column"]
    disagreements["evidence_note"] = (
        f"primary reference '{primary_column}' is a human benchmark, not "
        "adjudicated truth; confirming a model error needs per-annotator labels "
        "or independent adjudication, which are not available here"
    )
    disagreements = disagreements.sort_values(
        ["ordinal_distance", "reference", "id"], ascending=[False, True, True])
    _save(disagreements, results_dir / "t08_disagreements.csv")

    n_disagreements = int(len(disagreements))
    check("disagreement_count_reconciles", n_disagreements == n - n_correct,
          f"{n_disagreements} disagreement(s) = {n} - {n_correct}")

    category_rows = []
    for index, name in enumerate(classes):
        errors_from_class = matrix[index].copy()
        errors_from_class[index] = 0
        total_errors = int(errors_from_class.sum())
        entry = {
            "reference_class": name,
            "n": int(human_counts[index]),
            "n_correct": int(matrix[index, index]),
            "n_disagreements": total_errors,
            "recall": per_class[index]["recall"],
            "recall_ci_low": intervals.wilson(matrix[index, index],
                                              human_counts[index], level)[0],
            "recall_ci_high": intervals.wilson(matrix[index, index],
                                               human_counts[index], level)[1],
        }
        for target_index, target in enumerate(classes):
            entry[f"disagreements_to_{target}"] = int(errors_from_class[target_index])
        entry["most_common_error_target"] = (
            classes[int(np.argmax(errors_from_class))] if total_errors else "")
        category_rows.append(entry)
    _save(pd.DataFrame(category_rows), results_dir / "t09_category_summary.csv")

    status_rows = [
        {"status": "confirmed_model_error", "n": 0,
         "definition": "disagreement established as a model mistake by adjudication",
         "confirmable_with_this_data": False,
         "evidence_required": "per-annotator labels or independent adjudication"},
        {"status": "unconfirmed_disagreement", "n": n_disagreements,
         "definition": "prediction differs from the primary reference label",
         "confirmable_with_this_data": False,
         "evidence_required": "the same evidence as above; not estimable here"},
        {"status": "agreement", "n": n_correct,
         "definition": "prediction equals the primary reference label",
         "confirmable_with_this_data": True,
         "evidence_required": "none"},
    ]
    _save(pd.DataFrame(status_rows), results_dir / "t10_error_status.csv")

    # ---- 5. log, metadata, manifest -------------------------------------
    log("")
    log("== Results ==")
    log(f"n analysed: {n} | correct: {n_correct} | disagreements: {n_disagreements}")
    log(f"accuracy: {accuracy:.6f} (95% Wilson {acc_low:.6f}-{acc_high:.6f})")
    log(f"balanced accuracy: {core['balanced_accuracy']:.6f} "
        f"(bootstrap {boot_core['balanced_accuracy']['low']:.6f}-"
        f"{boot_core['balanced_accuracy']['high']:.6f})")
    log(f"kappa unweighted: {core['kappa_unweighted']:.6f} "
        f"(bootstrap {boot_core['kappa_unweighted']['low']:.6f}-"
        f"{boot_core['kappa_unweighted']['high']:.6f})")
    if weighted_requested:
        log(f"kappa quadratic: {core['kappa_quadratic']:.6f} "
            f"(bootstrap {boot_core['kappa_quadratic']['low']:.6f}-"
            f"{boot_core['kappa_quadratic']['high']:.6f})")
    log(f"ordinal buckets: {ordinal['buckets']}")
    for row in per_rows:
        log(f"  {row['class']:<8} n={row['n_reference']:<3} "
            f"precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f}")

    log("")
    log("== Method notes ==")
    log("- Confusion matrix orientation: rows = human reference, columns = AI.")
    log("- Accuracy and recall CIs: Wilson score. Kappa, balanced accuracy, "
        "precision and F1: seeded percentile bootstrap.")
    reference_note = (cfg.get("report", {}).get("reference_description") or "").strip()
    log("- Reference: " + (reference_note or "a human benchmark"))
    log("- The reference is a benchmark label, not an adjudicated gold standard. "
        "Where per-annotator labels are not released, human-human kappa is not "
        "computable; see t13_reference_agreement.csv for any secondary reference.")
    log("- Disagreements are not confirmed errors; see t10_error_status.csv.")
    n_not_ok = int(stats["excluded_not_ok"])
    if n_not_ok:
        log(f"- {n_not_ok} failed prediction(s) excluded from the metrics "
            "(status != 'ok'); never imputed. The two abstention policies differ "
            "only for failures; the exclusion is recorded in t01_join_and_sample.csv.")
    else:
        log("- 0 failed predictions in this run, so both abstention policies "
            "coincide; the exclusion count is recorded in t01_join_and_sample.csv.")
    log("- Ordinal distance is a severity proxy (adjacent vs opposite), not evidence "
        "about which label is right.")
    log("- No causal claims: these are associations between predicted and "
        "reference labels on the analysed sample.")
    log("- Transparency: aggregate benchmark class counts were seen before "
        "classification (to report the sampling frame). No per-row reference value "
        "entered the prompt or the classifier, and this evaluation still freezes "
        "predictions before opening the benchmark.")

    failed = [entry["check"] for entry in checks if not entry["ok"]]
    status = "PASS" if not failed else "FAIL"
    log("")
    log(f"checks passed: {len(checks) - len(failed)}/{len(checks)}")
    log(f"evaluation status: {status}" + (f"; failed: {failed}" if failed else ""))

    metadata = {
        "finished_at": now_iso(),
        "tool": {"name": "general-agent", "version": __version__},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_path": cfg["_meta"]["config_path"],
        "config_sha256": cfg["_meta"]["config_sha256"],
        "seed": seed,
        "ci_level": level,
        "bootstrap_draws": draws,
        "classes": classes,
        "ordinal": bool(cfg["labels"].get("ordered")),
        "weighted_kappa_reported": weighted_requested,
        "freeze": freeze,
        "benchmark": {
            "path": str(sealed_path),
            "sha256": file_sha256(sealed_path),
            "read_at": benchmark_read_at,
            "source": "sealed file restricted to the sampled IDs",
        },
        "row_counts": stats,
        "checks": {"passed": len(checks) - len(failed), "total": len(checks),
                   "failed": failed, "status": status},
        "tables": [
            "t01_join_and_sample.csv", "t02_accuracy.csv", "t03_kappa.csv",
            "t04_confusion_matrix.csv", "t05_per_class_metrics.csv",
            "t06_class_distribution.csv", "t07_ordinal_errors.csv",
            "t08_disagreements.csv", "t09_category_summary.csv",
            "t10_error_status.csv",
        ],
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    (results_dir / "evaluation_log.txt").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")

    update_manifest(out_dir, "05_freeze", {
        "frozen_at": freeze["frozen_at"],
        "source_sha256": freeze["source_sha256"],
        "frozen_sha256": freeze["frozen_sha256"],
        "n_rows": freeze["n_rows"],
        "status_counts": freeze["status_counts"],
    })
    update_manifest(out_dir, "05_evaluate", {
        "finished_at": metadata["finished_at"],
        "benchmark_read_at": benchmark_read_at,
        "row_counts": stats,
        "checks": metadata["checks"],
        "results_dir": str(results_dir),
        "tables": metadata["tables"],
        "seed": seed,
        "bootstrap_draws": draws,
        "accuracy": accuracy,
        "kappa_unweighted": core["kappa_unweighted"],
        "kappa_quadratic": core["kappa_quadratic"] if weighted_requested else None,
    })

    log(f"results written to: {results_dir}")
    print(f"finished: {now_iso()}")
    return 0 if status == "PASS" else 1
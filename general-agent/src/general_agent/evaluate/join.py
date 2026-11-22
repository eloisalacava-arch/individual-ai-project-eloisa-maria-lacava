"""Join frozen predictions to the sealed reference, with reconciliation checks.

Every check is recorded with a name, a pass/fail, and a detail string. Row
counts are printed for every merge step (AGENTS.md), and nothing is dropped
silently.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..discover import map_labels
from ..io import read_table, normalize_ids


class JoinError(RuntimeError):
    """Raised when the join cannot be reconciled."""


def join_frame(frozen: pd.DataFrame, benchmark: pd.DataFrame, cfg: dict,
               log) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Return (joined, stats, checks). ``log`` is a callable for row accounting."""
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    benchmark_cfg = cfg["benchmark"]
    # The sealed file uses the canonical ID column "id" written by stage 02,
    # whatever the source file called it. The reference label keeps its name.
    id_column = "id"
    label_column = benchmark_cfg["label_column"]

    missing_columns = [c for c in (id_column, label_column) if c not in benchmark.columns]
    if missing_columns:
        raise JoinError(
            f"sealed benchmark is missing columns {missing_columns}; expected "
            f"canonical 'id' plus the configured reference '{label_column}'"
        )

    if label_column == id_column:
        raise JoinError("benchmark label column is also the ID column")

    frozen = frozen.copy()
    frozen["id"] = normalize_ids(frozen["id"])
    # Rename the prediction before merging: the reference column may legitimately
    # be called "label" too, and a silent _x/_y suffix would corrupt the join.
    frozen = frozen.rename(columns={"label": "predicted"})
    benchmark = benchmark.copy()
    benchmark["id"] = normalize_ids(benchmark["id"])

    check("frozen_nonzero_rows", len(frozen) > 0, f"{len(frozen)} rows")
    check("benchmark_nonzero_rows", len(benchmark) > 0, f"{len(benchmark)} rows")
    check("frozen_ids_unique", not frozen["id"].duplicated().any(),
          f"{int(frozen['id'].duplicated().sum())} duplicate ID(s) of {len(frozen)}")
    check("benchmark_ids_unique", not benchmark["id"].duplicated().any(),
          f"{int(benchmark['id'].duplicated().sum())} duplicate ID(s) of {len(benchmark)}")

    frozen_ids = set(frozen["id"])
    benchmark_ids = set(benchmark["id"])
    only_frozen = sorted(frozen_ids - benchmark_ids)
    only_benchmark = sorted(benchmark_ids - frozen_ids)
    check("every_prediction_has_a_reference", not only_frozen,
          f"{len(only_frozen)} prediction ID(s) without a benchmark row"
          + (f" (examples {only_frozen[:3]})" if only_frozen else ""))
    check("every_reference_has_a_prediction", not only_benchmark,
          f"{len(only_benchmark)} benchmark ID(s) without a prediction"
          + (f" (examples {only_benchmark[:3]})" if only_benchmark else ""))

    before = len(frozen)
    extra_columns = [c for c in benchmark_cfg.get("extra_label_columns", [])
                     if c in benchmark.columns]
    reference_frame = benchmark[["id", label_column] + extra_columns].rename(
        columns={label_column: "reference"})
    if extra_columns:
        reference_frame = reference_frame.rename(
            columns={c: f"extra_{c}" for c in extra_columns})
    try:
        joined = frozen.merge(reference_frame, on="id", how="inner",
                              validate="one_to_one")
    except ValueError as exc:
        raise JoinError(f"join is not one-to-one: {exc}") from exc
    log(f"rows: {before} -> {len(joined)} (reason: inner join on id; "
        f"{before - len(joined)} prediction row(s) dropped because no reference matched)")

    check("join_kept_every_row", len(joined) == len(frozen),
          f"{len(joined)} joined of {len(frozen)} frozen")

    after = len(benchmark)
    log(f"rows: {after} -> {len(joined)} (reason: reference rows restricted to matched IDs; "
        f"{after - len(joined)} excluded)")

    # Map raw reference values onto declared classes. The sealed file keeps the
    # source spelling (e.g. 0/1); label_map is the configured answer.
    classes = cfg_classes(cfg)
    label_map = benchmark_cfg.get("label_map")
    mapped, unmapped = map_labels(joined["reference"], label_map, classes)
    check("primary_reference_mappable", not unmapped,
          f"unmapped reference values: {unmapped}" if unmapped
          else f"mapped onto {classes}")
    if unmapped:
        raise JoinError(f"reference values cannot be mapped: {unmapped}")
    joined["reference"] = mapped
    for column in extra_columns:
        target = f"extra_{column}"
        mapped_extra, unmapped_extra = map_labels(joined[target], label_map, classes)
        joined[target] = mapped_extra
        check(f"extra_reference_mappable[{column}]", not unmapped_extra,
              f"unmapped values (excluded from that comparison): {unmapped_extra}"
              if unmapped_extra else "mapped onto the declared classes")

    # Status handling: scorable rows only, with the exclusion counted.
    n_total = len(joined)
    scorable = joined[joined["status"] == "ok"].copy()
    log(f"rows: {n_total} -> {len(scorable)} (reason: status == 'ok'; "
        f"{n_total - len(scorable)} row(s) excluded as failed predictions)")
    check("no_failed_predictions", n_total == len(scorable),
          f"{n_total - len(scorable)} non-ok prediction(s)"
          + ("" if n_total == len(scorable) else "; excluded, never imputed"))

    check("reference_domain",
          set(scorable["reference"]).issubset(set(classes)),
          f"reference values: {sorted(set(scorable['reference']))}")
    for column in extra_columns:
        values = set(scorable[f"extra_{column}"].dropna())
        out_of_domain = sorted(values - set(classes))
        check(f"extra_reference_domain[{column}]", not out_of_domain,
              f"values outside the declared classes: {out_of_domain}"
              if out_of_domain else f"{len(values)} value(s), all declared")

    joined = scorable
    stats = {
        "predictions_rows": int(len(frozen)),
        "benchmark_rows": int(len(benchmark)),
        "matched_rows": int(len(joined)),
        "only_in_predictions": len(only_frozen),
        "only_in_benchmark": len(only_benchmark),
        "dropped_predictions": int(before - len(joined)),
        "excluded_not_ok": int(n_total - len(scorable)),
        "analysed_rows": int(len(joined)),
        "extra_references": extra_columns,
    }
    return joined, stats, checks


def cfg_classes(cfg: dict) -> list[str]:
    return [entry["name"] for entry in cfg["labels"]["classes"]]


def load_benchmark(out_dir: Path) -> pd.DataFrame:
    """Read the sealed reference, never the full source benchmark.

    The sealed file is restricted to the sampled IDs. Reading the source would
    reintroduce 3,000+ unmatched rows and defeat the seal.
    """
    path = Path(out_dir) / "sealed" / "benchmark.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run stage 02 (blind) before evaluating"
        )
    return pd.read_csv(path, dtype="string", keep_default_na=False)
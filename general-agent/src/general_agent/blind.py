"""Stage 02 - blind input and sealed benchmark.

The two files are created together and never in the same directory:

  blind/blind.csv        exactly (id, text) - all the classifier may read
  sealed/benchmark.csv   IDs, reference labels, and optional subgroup metadata

The sealed directory is chmod 000 for the duration of a classification run and
restored afterwards. That is a practical barrier against a same-user process
reading it, not a sandbox; see SPEC.md section 3 for the limits of each layer.

Sampling is seeded and unstratified. Stratifying would fix the class
distribution and inflate kappa, which depends on the marginals.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .config import class_names, load_config
from .discover import map_labels, observation_text
from .io import normalize_ids, read_table, write_json_atomic
from .provenance import file_sha256, now_iso, update_manifest
from .sealing import unlock_sealed

BLIND_COLUMNS = ["id", "text"]


class BlindError(RuntimeError):
    """Raised when a stage precondition is not met."""


def select_indices(frame: pd.DataFrame, mapping: dict, sample_cfg: dict,
                   seed: int) -> list[int]:
    """Choose rows in a deterministic, documented order.

    ``n`` absent/None means every row is used (still shuffled, so the
    classifier's input order does not follow the source file). ``unique_text``
    only applies when a subset is drawn: asking for "all rows" is never
    silently reduced.
    """
    texts = observation_text(frame, mapping).astype("string").tolist()
    n = sample_cfg.get("n")
    method = sample_cfg.get("method", "random")
    unique_text = bool(sample_cfg.get("unique_text", True))

    order = list(range(len(texts)))
    if method == "random":
        random.Random(seed).shuffle(order)

    if not n:
        return order

    selected: list[int] = []
    seen: set[str] = set()
    for index in order:
        text = texts[index]
        if unique_text and text in seen:
            continue
        seen.add(text)
        selected.append(index)
        if len(selected) >= n:
            break
    return selected


def reconcile_frames(cfg: dict, mapping: dict, dataset: pd.DataFrame,
                     benchmark: pd.DataFrame, emit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise the ID and label policies the audit validated in memory.

    The audit applies ``on_id_mismatch`` and ``on_unknown_label`` to in-memory
    copies and writes no derived dataset (SPEC section 7, decision 10). Stage 02
    re-reads the source files, so without reapplying the same restrictions the
    blind input and the sealed benchmark can refer to different row sets: the
    ID intersection is never taken, and a row whose label is excluded is still
    sampled. Every restriction is reported through ``emit`` with before/after
    counts and a reason.

    Only the primary reference drives exclusion here; extra reference columns
    are evaluated separately at stage 05 and never remove a row from the blind
    input.
    """
    classes = class_names(cfg)
    dataset_ids = normalize_ids(dataset[mapping["id_column"]])
    benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]])

    only_dataset = sorted(set(dataset_ids.dropna()) - set(benchmark_ids.dropna()))
    only_benchmark = sorted(set(benchmark_ids.dropna()) - set(dataset_ids.dropna()))
    if only_dataset or only_benchmark:
        if cfg["benchmark"]["on_id_mismatch"] == "inner_join":
            common = set(dataset_ids.dropna()) & set(benchmark_ids.dropna())
            before = len(dataset)
            dataset = dataset[dataset_ids.isin(common)].reset_index(drop=True)
            emit("dataset restricted to matched IDs", before, len(dataset),
                 f"{len(only_dataset)} dataset ID(s) with no benchmark row")
            before = len(benchmark)
            benchmark = benchmark[benchmark_ids.isin(common)].reset_index(drop=True)
            emit("benchmark restricted to matched IDs", before, len(benchmark),
                 f"{len(only_benchmark)} benchmark ID(s) with no dataset row")
            dataset_ids = normalize_ids(dataset[mapping["id_column"]])
            benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]])
        else:
            raise BlindError(
                f"ID sets differ ({len(only_dataset)} dataset-only, "
                f"{len(only_benchmark)} benchmark-only) and on_id_mismatch is "
                "'error'; fix the IDs or set benchmark.on_id_mismatch to inner_join"
            )

    primary = mapping["label_column"]
    mapped, unmapped = map_labels(
        benchmark[primary], cfg["benchmark"].get("label_map"), classes
    )
    if unmapped:
        if cfg["benchmark"]["on_unknown_label"] == "error":
            raise BlindError(
                f"benchmark label column '{primary}' has values {unmapped} outside "
                f"the declared classes {classes} and on_unknown_label is 'error'; "
                "map them or set benchmark.on_unknown_label to exclude"
            )
        keep = mapped.notna().to_numpy()
        before = len(benchmark)
        benchmark = benchmark[keep].reset_index(drop=True)
        benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]])
        emit("benchmark rows without a usable primary label", before, len(benchmark),
             f"{int((~keep).sum())} row(s) unmapped under policy 'exclude'")

        # The dataset must lose the same IDs, or the blind input would contain a
        # row the sealed benchmark can no longer score.
        scorable = set(benchmark_ids.dropna())
        before = len(dataset)
        keep_dataset = dataset_ids.isin(scorable).to_numpy()
        dataset = dataset[keep_dataset].reset_index(drop=True)
        dataset_ids = normalize_ids(dataset[mapping["id_column"]])
        emit("dataset restricted to scorable IDs", before, len(dataset),
             f"{int((~keep_dataset).sum())} row(s) whose benchmark label was excluded")

    return dataset, benchmark


def _load_decisions(out_dir: Path) -> dict:
    path = out_dir / "decisions.json"
    if not path.exists():
        raise BlindError(
            f"{path} is missing: run stage 01 (audit) before stage 02 (blind)"
        )
    decisions = json.loads(path.read_text(encoding="utf-8"))
    if decisions.get("status") != "resolved":
        raise BlindError(
            "decisions.json status is not 'resolved'; answer the audit questions first"
        )
    return decisions


def run_blind(config_path: str | Path, force: bool = False,
              unlock_only: bool = False) -> int:
    cfg = load_config(config_path)
    out_dir = Path(cfg["_meta"]["output_dir"])

    if unlock_only:
        restored = unlock_sealed(out_dir)
        print(f"sealed directory permissions restored: {restored}")
        return 0

    decisions = _load_decisions(out_dir)
    mapping = decisions["resolved_mapping"]
    classes = class_names(cfg)

    blind_dir = out_dir / "blind"
    sealed_dir = out_dir / "sealed"
    blind_path = blind_dir / "blind.csv"
    sealed_path = sealed_dir / "benchmark.csv"
    if (blind_path.exists() or sealed_path.exists()) and not force:
        raise BlindError(
            f"{blind_dir} or {sealed_dir} already holds a sealed run; "
            "pass --force only if you intend to discard it"
        )

    unlock_sealed(out_dir)
    blind_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    print("script: general-agent blind (stage 02)")
    print(f"started: {now_iso()}")
    print(f"run_id: {cfg['run']['run_id']}   seed: {cfg['run']['seed']}")

    dataset = read_table(cfg["dataset"]["path"], cfg["dataset"]["format"],
                         cfg["dataset"]["delimiter"], cfg["dataset"]["encoding"])
    benchmark = read_table(cfg["benchmark"]["path"], cfg["benchmark"]["format"],
                           cfg["benchmark"]["delimiter"], cfg["benchmark"]["encoding"])
    rows_dataset_read = len(dataset)
    rows_benchmark_read = len(benchmark)
    print(f"rows read from dataset: {rows_dataset_read} (no filter)")
    print(f"rows read from benchmark: {rows_benchmark_read} (no filter)")

    reconciliation: list[dict] = []

    def emit(step: str, before: int, after: int, reason: str) -> None:
        dropped = before - after
        reconciliation.append({
            "step": step, "rows_before": before, "rows_after": after,
            "rows_dropped": dropped, "reason": reason,
        })
        print(f"rows: {before} -> {after} ({dropped} dropped; reason: {reason})")

    dataset, benchmark = reconcile_frames(cfg, mapping, dataset, benchmark, emit)
    print(f"rows after audit policies: dataset {len(dataset)}, "
          f"benchmark {len(benchmark)}")

    sampled = select_indices(dataset, mapping, cfg["sample"], cfg["run"]["seed"])
    print(f"rows: {len(dataset)} -> {len(sampled)} "
          f"(reason: sample method={cfg['sample'].get('method')}, "
          f"n={cfg['sample'].get('n')}, unique_text={cfg['sample'].get('unique_text')}, "
          f"seed={cfg['run']['seed']}, unstratified)")

    sample_frame = dataset.iloc[sampled]
    ids = normalize_ids(sample_frame[mapping["id_column"]])
    texts = observation_text(sample_frame, mapping).astype("string")
    if ids.duplicated().any():
        raise BlindError("sampled IDs are not unique; refusing to build a blind input")
    if texts.str.strip().eq("").any():
        raise BlindError("sampled observation text is blank")

    benchmark_ids = normalize_ids(benchmark[mapping["id_column_benchmark"]])
    keep = benchmark_ids.isin(set(ids))
    sealed_frame = benchmark.loc[keep].copy()
    print(f"rows: {len(benchmark)} -> {len(sealed_frame)} "
          f"(reason: restricted to the {len(ids)} sampled IDs; "
          f"{int((~keep).sum())} unmatched benchmark rows excluded)")
    if len(sealed_frame) != len(ids):
        raise BlindError(
            f"{len(ids)} sampled IDs but {len(sealed_frame)} benchmark rows; "
            "the audit's ID check should have caught this"
        )

    sealed_frame["id"] = benchmark_ids[keep].to_numpy()
    reference_columns = [mapping["label_column"]] + list(mapping.get("extra_label_columns", []))
    subgroup_columns = [c for c in cfg["evaluation"].get("subgroup_columns", [])
                        if c in sealed_frame.columns]
    sealed_columns = ["id"] + reference_columns + subgroup_columns
    sealed_frame = sealed_frame[sealed_columns]

    # Follow the blind order so row i of the sealed file corresponds to row i of
    # the blind file. The join is by ID anyway; this only makes review easier.
    sealed_frame = sealed_frame.set_index("id")
    absent = [identifier for identifier in ids.tolist() if identifier not in sealed_frame.index]
    if absent:
        raise BlindError(f"{len(absent)} sampled ID(s) missing from the sealed benchmark")
    sealed_frame = sealed_frame.loc[ids.tolist()].reset_index()

    blind_frame = pd.DataFrame({
        "id": ids.to_numpy(),
        "text": texts.to_numpy(),
    })[BLIND_COLUMNS]

    blind_dir.mkdir(parents=True, exist_ok=True)
    blind_frame.to_csv(blind_path, index=False)
    sealed_frame.to_csv(sealed_path, index=False)

    # Structural assertions: the classifier's directory holds one file only.
    entries = sorted(p.name for p in blind_dir.iterdir())
    if entries != ["blind.csv"]:
        raise BlindError(f"blind directory must contain only blind.csv, found {entries}")
    if Path(sealed_dir).resolve() == Path(blind_dir).resolve():
        raise BlindError("sealed and blind directories must not be the same")

    counts = {}
    for column in reference_columns:
        if column in sealed_frame.columns:
            counts[column] = {
                str(key): int(value)
                for key, value in sealed_frame[column].value_counts().to_dict().items()
            }
    info = {
        "run_id": cfg["run"]["run_id"],
        "created_at": now_iso(),
        "seed": cfg["run"]["seed"],
        "sample": {
            "method": cfg["sample"].get("method"),
            "requested_n": cfg["sample"].get("n"),
            "selected_n": int(len(ids)),
            "unique_text": cfg["sample"].get("unique_text"),
            "stratified": False,
        },
        "class_names": classes,
        "reference_columns": reference_columns,
        "subgroup_columns": subgroup_columns,
        "reference_distributions": counts,
        "n_selected": int(len(ids)),
        "reconciliation": reconciliation,
        "rows_dataset_read": int(rows_dataset_read),
        "rows_benchmark_read": int(rows_benchmark_read),
        "rows_dataset_after_policies": int(len(dataset)),
        "rows_benchmark_after_policies": int(len(benchmark)),
        "source_dataset": str(cfg["dataset"]["path"]),
        "source_benchmark": str(cfg["benchmark"]["path"]),
        "blind_sha256": file_sha256(blind_path),
        "sealed_sha256": file_sha256(sealed_path),
    }
    write_json_atomic(sealed_dir / "blind_info.json", info)
    write_json_atomic(blind_dir.parent / "blind_public_info.json", {
        key: info[key] for key in
        ("run_id", "created_at", "seed", "sample", "class_names", "n_selected",
         "blind_sha256")
    })

    update_manifest(out_dir, "02_blind", {
        "finished_at": now_iso(),
        "blind_path": str(blind_path),
        "blind_sha256": info["blind_sha256"],
        "sealed_sha256": info["sealed_sha256"],
        "rows_in": int(rows_dataset_read),
        "rows_selected": int(len(ids)),
        "rows_benchmark_in": int(rows_benchmark_read),
        "rows_benchmark_sealed": int(len(sealed_frame)),
        "rows_dataset_after_policies": int(len(dataset)),
        "rows_benchmark_after_policies": int(len(benchmark)),
        "reconciliation": reconciliation,
        "sample": info["sample"],
        "class_names": classes,
        "reference_columns": reference_columns,
        "subgroup_columns": subgroup_columns,
        "blind_dir_entries": entries,
    })

    print(f"blind written: {blind_path} ({len(blind_frame)} rows, columns {BLIND_COLUMNS})")
    print(f"sealed written: {sealed_path} ({len(sealed_frame)} rows, columns {sealed_columns})")
    print(f"blind sha256: {info['blind_sha256']}")
    print(f"sealed sha256: {info['sealed_sha256']}")
    print("sealed directory is separate from the blind directory and is locked during classification")
    print(f"finished: {now_iso()}")
    return 0
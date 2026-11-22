#!/usr/bin/env python
"""Prepare the IMDb case-study benchmark file (example adapter, not core).

The generic agent needs two separate files: a dataset (ID + observation text)
and a benchmark (ID + human label). The existing IMDb project keeps the human
annotations and the star-derived dataset label in two different files under
``data/``. This script joins them into ``inputs/imdb_benchmark.csv``.

It is deliberately an *adapter*: a new study normally starts from a single
benchmark file and does not need a script like this.

Reads (never modifies):
    ../data/annotations.csv              human annotation, one row per review
    ../data/derived/sample_mapping.csv   review_id -> star-derived label (0/1)

Writes:
    inputs/imdb_benchmark.csv

Exit codes: 0 ok, 1 validation failure.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

AGENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AGENT_ROOT.parent
ANNOTATIONS = PROJECT_ROOT / "data" / "annotations.csv"
MAPPING = PROJECT_ROOT / "data" / "derived" / "sample_mapping.csv"
OUTPUT = AGENT_ROOT / "inputs" / "imdb_benchmark.csv"

OUTPUT_COLUMNS = [
    "review_id",
    "human_sentiment",
    "human_sarcasm",
    "human_mixed",
    "human_ambiguous",
    "human_complex",
    "human_straightforward",
    "uncertainty_sentiment",
    "imdb_label",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing inputs/imdb_benchmark.csv")
    args = parser.parse_args(argv)

    if OUTPUT.exists() and not args.force:
        print(f"refusing to overwrite {OUTPUT} without --force")
        return 1
    for path in (ANNOTATIONS, MAPPING):
        if not path.exists():
            print(f"ABORT: required input not found: {path}")
            return 1

    print("script: prepare_imdb_benchmark.py")
    print(f"annotations: {ANNOTATIONS}  sha256 {sha256(ANNOTATIONS)[:16]}...")
    print(f"mapping:     {MAPPING}  sha256 {sha256(MAPPING)[:16]}...")

    annotations = pd.read_csv(ANNOTATIONS)
    mapping = pd.read_csv(MAPPING)
    print(f"rows read from annotations.csv: {len(annotations)} (no filter)")
    print(f"rows read from sample_mapping.csv: {len(mapping)} (no filter)")
    print(f"annotations columns: {list(annotations.columns)}")
    print(f"mapping columns: {list(mapping.columns)}")

    annotation_ids = set(annotations["review_id"].astype(str).str.strip())
    mapping_ids = set(mapping["review_id"].astype(str).str.strip())
    assert len(annotations) == len(annotation_ids), "duplicate review_id in annotations"
    assert len(mapping) == len(mapping_ids), "duplicate review_id in sample_mapping"
    only_annotations = sorted(annotation_ids - mapping_ids)
    only_mapping = sorted(mapping_ids - annotation_ids)
    if only_annotations or only_mapping:
        print(f"ABORT: ID mismatch: {len(only_annotations)} annotation-only, "
              f"{len(only_mapping)} mapping-only")
        return 1
    print(f"ID match: {len(annotation_ids)} annotations <-> {len(mapping_ids)} mapping rows "
          "(0 dropped, 0 duplicated)")

    merged = annotations.merge(mapping[["review_id", "label"]], on="review_id",
                               how="inner", validate="one_to_one")
    print(f"rows after inner join: {len(merged)} (of {len(annotations)}; "
          f"{len(annotations) - len(merged)} dropped: none expected)")

    bad_human = sorted(set(merged["sentiment"].dropna()) - {"positive", "negative"})
    bad_imdb = sorted(set(merged["label"].dropna()) - {0, 1})
    if bad_human or bad_imdb:
        print(f"ABORT: unexpected label values human={bad_human} imdb={bad_imdb}")
        return 1

    inconsistent = merged[
        (merged["straightforward"] == 1)
        != (merged[["sarcasm", "mixed", "ambiguous", "complex"]].sum(axis=1) == 0)
    ]
    if len(inconsistent):
        print(f"ABORT: {len(inconsistent)} row(s) where straightforward is not "
              "the derivation of the four flags")
        return 1

    out = pd.DataFrame({
        "review_id": merged["review_id"].astype(str).str.strip(),
        "human_sentiment": merged["sentiment"].astype(str).str.strip(),
        "human_sarcasm": merged["sarcasm"].astype(int),
        "human_mixed": merged["mixed"].astype(int),
        "human_ambiguous": merged["ambiguous"].astype(int),
        "human_complex": merged["complex"].astype(int),
        "human_straightforward": merged["straightforward"].astype(int),
        "uncertainty_sentiment": merged["uncertainty_sentiment"].astype(int),
        "imdb_label": merged["label"].astype(int),
    })[OUTPUT_COLUMNS]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"rows written: {len(out)} -> {OUTPUT}")
    print(f"columns: {list(out.columns)}")
    print(f"sha256(output): {sha256(OUTPUT)}")
    print("human_sentiment counts: "
          f"{out['human_sentiment'].value_counts().to_dict()}")
    print("imdb_label counts (0=negative, 1=positive, per data/README.md): "
          f"{out['imdb_label'].value_counts().sort_index().to_dict()}")
    print("sources were opened read-only; nothing under data/ was modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""Fetch and convert a public benchmark into the agent's two-file layout.

Currently supports the Financial PhraseBank v1.0 (Malo et al. 2014), a
3-class sentence-level sentiment corpus annotated by a panel of 16 finance
professionals, released at four agreement levels.

Why download the archive directly instead of ``datasets.load_dataset``:
the Hub copy is a loading *script*, which modern ``datasets`` versions only run
with ``trust_remote_code=True``. Downloading the official archive and parsing
its text files avoids executing remote code and avoids a heavy dependency.

Licence: CC BY-NC-SA 3.0 (non-commercial, share-alike). The data is therefore
written only under the gitignored ``inputs/`` directory and must not be
redistributed with this agent.

Outputs (all under --out):
    dataset.csv    id,text            <- what the classifier may read
    benchmark.csv  id,label           <- the human panel's majority label
    source.json    provenance, hashes, counts, licence, citation

Exit codes: 0 ok, 1 failure.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
LICENCE = "CC BY-NC-SA 3.0 (non-commercial, share-alike)"
CITATION = (
    "Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). "
    "Good debt or bad debt: Detecting semantic orientations in economic texts. "
    "Journal of the Association for Information Science and Technology, 65."
)
CLASSES = ("negative", "neutral", "positive")
AGREEMENT_FILES = {
    "50": "Sentences_50Agree.txt",
    "66": "Sentences_66Agree.txt",
    "75": "Sentences_75Agree.txt",
    "all": "Sentences_AllAgree.txt",
}
EXPECTED_ROWS = {"50": 4846, "66": 4217, "75": 3453, "all": 2264}


def _download(url: str, cache: Path) -> bytes:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print(f"using cached archive: {cache}")
        return cache.read_bytes()
    print(f"downloading: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "general-agent/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    cache.write_bytes(payload)
    print(f"cached archive: {cache} ({len(payload)} bytes)")
    return payload


def _extract_member(payload: bytes, filename: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        candidates = [name for name in archive.namelist() if name.endswith(filename)]
        if not candidates:
            raise FileNotFoundError(
                f"{filename} not found in archive; members: {archive.namelist()}"
            )
        if len(candidates) > 1:
            raise ValueError(f"ambiguous archive member for {filename}: {candidates}")
        print(f"archive member: {candidates[0]}")
        return archive.read(candidates[0])


def _parse(raw: bytes) -> list[tuple[str, str]]:
    """Each line is ``sentence@label``; split on the LAST '@' only."""
    text = raw.decode("iso-8859-1")
    rows: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.rstrip("\r")
        if not line.strip():
            continue
        if "@" not in line:
            raise ValueError(f"line {number} has no '@' separator: {line[:80]!r}")
        sentence, label = line.rsplit("@", 1)
        rows.append((sentence.strip(), label.strip()))
    return rows


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", default="75", choices=sorted(AGREEMENT_FILES),
                        help="annotator-agreement subset (default: 75, the usual convention)")
    parser.add_argument("--out", default="inputs/financial_phrasebank/75agree",
                        help="output directory (relative to the agent folder)")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    if (out_dir / "dataset.csv").exists() and not args.force:
        print(f"refusing to overwrite {out_dir} without --force")
        return 1

    print("script: fetch_dataset.py (financial_phrasebank)")
    print(f"licence: {LICENCE}")
    print(f"agreement subset: {args.agreement}")

    cache = Path(__file__).resolve().parents[1] / "inputs" / "_cache" / "FinancialPhraseBank-v1.0.zip"
    payload = _download(URL, cache)
    archive_sha = sha256_bytes(payload)
    print(f"archive sha256: {archive_sha}")

    raw = _extract_member(payload, AGREEMENT_FILES[args.agreement])
    pairs = _parse(raw)
    print(f"rows parsed: {len(pairs)}")
    expected = EXPECTED_ROWS[args.agreement]
    if len(pairs) != expected:
        print(f"ABORT: expected {expected} rows for agreement={args.agreement}, got {len(pairs)}")
        return 1

    bad_labels = sorted({label for _, label in pairs} - set(CLASSES))
    if bad_labels:
        print(f"ABORT: undeclared labels {bad_labels}; declared classes are {CLASSES}")
        return 1

    prefix = f"fpb{args.agreement}"
    ids = [f"{prefix}_{index:05d}" for index in range(len(pairs))]
    dataset_rows = [(ids[i], pairs[i][0]) for i in range(len(pairs))]
    benchmark_rows = [(ids[i], pairs[i][1]) for i in range(len(pairs))]

    texts = [sentence for sentence, _ in pairs]
    duplicate_texts = len(texts) - len(set(texts))
    counts: dict[str, int] = {}
    for _, label in pairs:
        counts[label] = counts.get(label, 0) + 1
    print(f"duplicate sentences: {duplicate_texts} (kept; reported, not dropped)")
    print(f"label distribution: {dict(sorted(counts.items()))}")

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "dataset.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("id,text\n")
        for identifier, text in dataset_rows:
            escaped = text.replace('"', '""')
            fh.write(f'{identifier},"{escaped}"\n')

    benchmark_path = out_dir / "benchmark.csv"
    with benchmark_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("id,label\n")
        for identifier, label in benchmark_rows:
            fh.write(f"{identifier},{label}\n")

    print(f"rows written to {dataset_path}: {len(dataset_rows)} (columns id,text)")
    print(f"rows written to {benchmark_path}: {len(benchmark_rows)} (columns id,label)")

    source = {
        "source": "takala/financial_phrasebank (Hugging Face), FinancialPhraseBank-v1.0.zip",
        "url": URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": archive_sha,
        "archive_member": AGREEMENT_FILES[args.agreement],
        "agreement_subset": args.agreement,
        "licence": LICENCE,
        "licence_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "citation": CITATION,
        "encoding": "iso-8859-1",
        "parse_rule": "line.rsplit('@', 1) -> (sentence, label)",
        "id_rule": f"{prefix}_<zero-padded file index> (deterministic for this archive)",
        "n_rows": len(pairs),
        "label_counts": dict(sorted(counts.items())),
        "duplicate_sentences": duplicate_texts,
        "dataset_csv_sha256": sha256_bytes(dataset_path.read_bytes()),
        "benchmark_csv_sha256": sha256_bytes(benchmark_path.read_bytes()),
    }
    (out_dir / "source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"provenance written: {out_dir / 'source.json'}")

    if args.agreement == "75" and out_dir != Path(__file__).resolve().parents[1] / "inputs/financial_phrasebank/75agree":
        print(f"note: case2_phrasebank.yaml expects inputs/financial_phrasebank/75agree; "
              f"this run wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
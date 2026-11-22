#!/usr/bin/env python
"""Build the publication-safe subset of reports and provenance.

Reads original files only. Writes only under ``general-agent/publication``.
No model calls, no changes to the original runs, reports, data or comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parents[2]
PUB = AGENT_ROOT / "publication"
RUNS = AGENT_ROOT / "runs"
REPORTS = AGENT_ROOT / "reports"
COMPARISON = AGENT_ROOT / "comparison"
DATA = REPO_ROOT / "data"
PROJECT_PREFIX = str(REPO_ROOT)
HOME_PREFIX = str(Path.home())

sys.path.insert(0, str(AGENT_ROOT / "src"))
from general_agent.config import load_config  # noqa: E402
from general_agent.report import figures as report_figures  # noqa: E402
from general_agent.report import pdf as report_pdf  # noqa: E402

RECORDS: list[dict[str, str]] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def record(original: Path, published: Path, change: str) -> None:
    RECORDS.append({
        "original": str(original.relative_to(REPO_ROOT)),
        "original_sha256": sha256(original),
        "published": str(published.relative_to(REPO_ROOT)),
        "published_sha256": sha256(published),
        "change": change,
    })


def copy_file(original: Path, published: Path, change: str = "byte-identical copy") -> None:
    ensure(published)
    shutil.copy2(original, published)
    record(original, published, change)


def write_text(original: Path, published: Path, text: str, change: str) -> None:
    ensure(published)
    published.write_text(text, encoding="utf-8")
    record(original, published, change)


def sanitize_study(markdown: str) -> str:
    lines = markdown.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "### Illustrative examples"]
    if len(starts) != 1:
        raise RuntimeError(f"expected one illustrative-examples heading, found {len(starts)}")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), None)
    if end is None:
        raise RuntimeError("no following section after illustrative examples")
    note = [
        "### Illustrative examples",
        "",
        "Illustrative examples are omitted from this public copy to avoid redistributing source text. "
        "The original examples remain in the author's working tree and are not part of this repository.",
        "",
        "Source for the aggregate qualitative tables: `results/qualitative/tables/q02_patterns.csv`, "
        "`q05_coverage.csv`, `q07_reason_contrast.csv`.",
        "",
    ]
    return "\n".join(lines[:start] + note + lines[end:]) + "\n"


def sanitize_qualitative(markdown: str) -> str:
    lines = markdown.splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.strip() == "## 4. Illustrative examples (disagreements)"]
    if len(starts) != 1:
        raise RuntimeError(f"expected one illustrative-examples heading, found {len(starts)}")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## 5.")), None)
    if end is None:
        raise RuntimeError("no following limitations section")
    note = [
        "## 4. Illustrative examples (disagreements)",
        "",
        "Illustrative examples are omitted from this public copy to avoid redistributing source text. "
        "The original examples remain in the author's working tree and are not part of this repository.",
        "",
    ]
    return "\n".join(lines[:start] + note + lines[end:]) + "\n"


def scan_forbidden(path: Path, include_filenames: bool = True) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    found = []
    if PROJECT_PREFIX in text:
        found.append("<project root path>")
    if HOME_PREFIX in text:
        found.append("<home path>")
    if include_filenames:
        for token in ["evidence:", "“", "”", "q04_examples.csv", "text_excerpt",
                      "selected.csv"]:
            if token in text:
                found.append(token)
    if found:
        raise RuntimeError(f"{path}: forbidden content remains: {found}")


def sanitize_string(value: str) -> str:
    return (value
            .replace(PROJECT_PREFIX, "<project-root>")
            .replace(HOME_PREFIX, "<home>"))


def sanitize_object(value):
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, list):
        return [sanitize_object(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_object(item) for key, item in value.items()}
    return value


def sanitized_json(original: Path, published: Path, redact_text_examples: bool = False) -> None:
    original_text = original.read_text(encoding="utf-8")
    data = json.loads(original_text)
    data = sanitize_object(data)
    if redact_text_examples:
        for profiles in (data.get("profiles") or {}).values():
            for profile in profiles:
                if profile.get("name") == "text":
                    profile["examples"] = ["<redacted: source text>"]
    published_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if redact_text_examples:
        change = "absolute paths rewritten; text examples redacted"
    elif published_text != original_text:
        change = "absolute paths rewritten"
    else:
        change = "byte-identical copy (no redaction needed)"
    write_text(original, published, published_text, change)


def sanitized_text(original: Path, published: Path) -> None:
    original_text = original.read_text(encoding="utf-8")
    published_text = sanitize_string(original_text)
    change = "absolute paths rewritten" if published_text != original_text else "byte-identical copy (no redaction needed)"
    write_text(original, published, published_text, change)


def study_pdf(run_id: str, config_path: Path, markdown_path: Path, pdf_path: Path) -> None:
    cfg = load_config(config_path)
    figures = {
        "confusion_matrix": report_figures.confusion_matrix_png(cfg),
        "per_class_recall": report_figures.per_class_recall_png(cfg),
    }
    report_pdf.markdown_to_pdf(
        markdown_path.read_text(encoding="utf-8"),
        pdf_path,
        title=cfg.get("report", {}).get("title") or run_id,
        footer=f"run {run_id}",
        hypotheses=True,
        figures=figures,
    )


def qualitative_pdf(run_id: str, markdown_path: Path, pdf_path: Path) -> None:
    report_pdf.markdown_to_pdf(
        markdown_path.read_text(encoding="utf-8"),
        pdf_path,
        title=f"Qualitative analysis — {run_id}",
        hypotheses=True,
    )


def build_reports() -> None:
    studies = [
        ("imdb_case_study", AGENT_ROOT / "configs/imdb_case_study.yaml", "IMDb case study"),
        ("phrasebank_75agree", AGENT_ROOT / "configs/case2_phrasebank.yaml", "Financial PhraseBank"),
    ]
    for run_id, config_path, label in studies:
        original = REPORTS / run_id / "study_report.md"
        published = PUB / "reports" / run_id / "study_report.md"
        sanitized = sanitize_study(original.read_text(encoding="utf-8"))
        write_text(original, published, sanitized,
                   "illustrative examples removed; all statistics retained")
        scan_forbidden(published)

        pdf_path = PUB / "reports" / run_id / "study_report.pdf"
        study_pdf(run_id, config_path, published, pdf_path)
        record(original.with_suffix(".pdf"), pdf_path,
               "regenerated from sanitized Markdown; figures retained")

        q_original = RUNS / run_id / "qualitative" / "narrative_report.md"
        q_published = PUB / "reports" / run_id / "qualitative_report.md"
        q_sanitized = sanitize_qualitative(q_original.read_text(encoding="utf-8"))
        write_text(q_original, q_published, q_sanitized,
                   "illustrative examples removed; aggregate findings retained")
        scan_forbidden(q_published)
        q_pdf = PUB / "reports" / run_id / "qualitative_report.pdf"
        qualitative_pdf(run_id, q_published, q_pdf)
        record(q_original, q_pdf, "regenerated from sanitized Markdown")

    comparison_md = COMPARISON / "report.md"
    comparison_pdf = REPORTS / "comparison" / "report.pdf"
    copy_file(comparison_md, PUB / "reports/comparison/report.md")
    copy_file(comparison_pdf, PUB / "reports/comparison/report.pdf")


def build_provenance() -> None:
    common = [
        ("manifest.json", "json"),
        ("audit.txt", "text"),
        ("decisions.json", "decisions"),
        ("blind_public_info.json", "json"),
        ("sealed/blind_info.json", "json"),
        ("pilot/pilot_report.json", "json"),
        ("results/evaluation_log.txt", "text"),
        ("results/run_metadata.json", "json"),
        ("results/predictions_freeze.json", "json"),
    ]
    per_run = {
        "imdb_case_study": common + [("predictions_import.json", "json")],
        "phrasebank_75agree": common + [("pilot/approval.json", "copy")],
    }
    for run_id, items in per_run.items():
        for relative, kind in items:
            original = RUNS / run_id / relative
            if not original.exists():
                print(f"skip missing: {original}")
                continue
            published = PUB / "provenance" / run_id / relative
            if kind == "json":
                sanitized_json(original, published)
            elif kind == "decisions":
                sanitized_json(original, published, redact_text_examples=True)
            elif kind == "text":
                sanitized_text(original, published)
            else:
                copy_file(original, published)
            if published.suffix in {".json", ".txt"}:
                scan_forbidden(published, include_filenames=False)

    sanitized_json(COMPARISON / "run_metadata.json",
                   PUB / "provenance/comparison/run_metadata.json")


def build_qualitative() -> None:
    root_files = ["settings.yaml", "approval.json", "selection_meta.json"]
    table_files = ["q00_selection.csv", "q02_patterns.csv", "q03_cooccurrence.csv",
                   "q05_coverage.csv", "q07_reason_contrast.csv"]
    for run_id in ("imdb_case_study", "phrasebank_75agree"):
        for name in root_files:
            original = RUNS / run_id / "qualitative" / name
            if not original.exists():
                print(f"skip missing: {original}")
                continue
            published = PUB / "qualitative" / run_id / name
            if original.suffix == ".json":
                sanitized_json(original, published)
            elif original.suffix == ".yaml":
                sanitized_text(original, published)
            else:
                copy_file(original, published)
        for name in table_files:
            original = RUNS / run_id / "qualitative" / "tables" / name
            if not original.exists():
                print(f"skip missing: {original}")
                continue
            copy_file(original, PUB / "qualitative" / run_id / "tables" / name)


def build_context() -> None:
    original = REPO_ROOT / "results" / "verification_findings.md"
    published = PUB / "context" / "verification_findings_public.md"
    text = re.sub(r"single blind annotator \([A-Z]{3}\)",
                  "single blind annotator (<annotator>)",
                  original.read_text(encoding="utf-8"))
    write_text(original, published, text, "annotator initials redacted")
    copy_file(REPO_ROOT / "results" / "agent_autonomous_decisions.md",
              PUB / "context" / "agent_autonomous_decisions.md")


def build_annotations() -> None:
    original = DATA / "annotations.csv"
    published = PUB / "data/annotations_public.csv"
    ensure(published)
    with original.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name for name in reader.fieldnames or [] if name != "annotator"]
        with published.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                writer.writerow({key: row[key] for key in fieldnames})
    record(original, published, "annotator column removed; all other values unchanged")


def build_manifest() -> None:
    lines = [
        "# Publication manifest",
        "",
        "This file records every publication copy, the original source, both SHA-256 hashes, "
        "and the nature of the change. Original files were not modified.",
        "",
        "| Original | Original SHA-256 | Published copy | Published SHA-256 | Change |",
        "|---|---|---|---|---|",
    ]
    for row in RECORDS:
        lines.append(
            f"| `{row['original']}` | `{row['original_sha256']}` | `{row['published']}` | "
            f"`{row['published_sha256']}` | {row['change']} |"
        )
    lines += [
        "",
        "## Redaction rules",
        "",
        "- Absolute authoring-machine paths were replaced with `<project-root>` and `<home>` placeholders.",
        "- `decisions.json` text examples were replaced with `<redacted: source text>`.",
        "- Illustrative examples were removed from the public study and qualitative reports.",
        "- `annotations_public.csv` omits the `annotator` column.",
        "- Annotator initials were removed from `context/verification_findings_public.md`.",
        "",
        "All statistical values, tables, figure inputs, conclusions and limitations are unchanged.",
        "",
    ]
    (PUB / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PUB.mkdir(parents=True, exist_ok=True)
    build_reports()
    build_provenance()
    build_qualitative()
    build_context()
    build_annotations()
    build_manifest()
    print(f"wrote {len(RECORDS)} publication copies")


if __name__ == "__main__":
    main()
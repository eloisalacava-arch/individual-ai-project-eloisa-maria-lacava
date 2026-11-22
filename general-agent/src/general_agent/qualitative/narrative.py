"""Compose the narrative report from tables and the statistical report.

Deterministic: no model call. Every number is formatted from a CSV under
``results/`` or ``qualitative/tables/``; the narrative never types a value.

The section builders are public so the per-study report composer can reuse the
same qualitative content without duplicating it. Rendered tables use readable
reason labels; the CSV tables keep the machine codes for reproducibility.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..provenance import now_iso
from . import prompts
from .schema import reason_label, reason_label_lower
from .settings import QualitativeError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_EXAMPLES = 6


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "not estimable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "not estimable"
    return f"{number:.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not estimable"
    if math.isnan(number):
        return "not estimable"
    return f"{number * 100:.{digits}f}%"


def _read(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise QualitativeError(f"{path} is missing; run the qualitative report step")
    return pd.read_csv(Path(path))


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return lines


# Readable column headings for the rendered tables. The CSV tables keep their
# machine column names for reproducibility; only the display copy is relabelled,
# so the long snake_case headers do not overflow a narrow PDF column.
HEADER_LABELS = {
    "reason": "Reason",
    "group": "Group",
    "n": "n",
    "share_of_explained": "Share of explained",
    "share_of_selected": "Share of selected",
    "top_directions": "Top directions",
    "share_disagreements": "Disagreements",
    "share_control": "Control",
    "difference_disagreements_minus_control":
        "Difference (disagreements minus control)",
    "n_selected": "Selected",
    "n_explained": "Explained",
    "n_failed": "Failed",
    "explained_share": "Explained share",
    "statuses": "Statuses",
    "step": "Step",
    "rows_before": "Rows before",
    "rows_after": "Rows after",
    "rows_dropped": "Rows dropped",
}


def _labelled(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy with readable headings and reason codes replaced by labels."""
    display = frame.copy()
    renamed = []
    mapping: dict[str, str] = {}
    for column in columns:
        target = HEADER_LABELS.get(column, column)
        renamed.append(target)
        if target != column:
            mapping[column] = target
    if "reason" in display.columns:
        display["reason"] = display["reason"].map(reason_label)
    if mapping:
        display = display.rename(columns=mapping)
    return display, renamed


def _portable(path: str, run_id: str) -> str:
    """A project-relative config path, or a run-id reference when not portable."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return ""


def scope_lines(out_dir: Path, settings: dict) -> list[str]:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    coverage = _read(tables / "q05_coverage.csv")
    selection = _read(tables / "q00_selection.csv")
    classes = settings["_meta"]["classes"]
    hashes = prompts.prompt_template_hashes(settings, classes)

    lines: list[str] = []
    lines.append(f"- Model that proposed explanations: `{settings['model']['name']}`.")
    lines.append(f"- Vocabulary version {settings.get('vocabulary_version', 1)}; "
                 f"prompt version {settings.get('prompt_version', 1)}.")
    lines.append(f"- Prompt-template hashes: disagreement "
                 f"`{hashes['disagreement'][:16]}`, agreement "
                 f"`{hashes['agreement'][:16]}`.")
    lines.append(f"- Seed {settings['sample']['seed']}; sampled "
                 f"{settings['sample']['n']} disagreements and "
                 f"{settings['sample']['control_n']} controls.")
    lines.append("")
    coverage_display, coverage_headers = _labelled(
        coverage, ["group", "n_selected", "n_explained", "n_failed",
                   "explained_share", "statuses"])
    lines.extend(_markdown_table(coverage_display, coverage_headers))
    lines.append("")
    lines.append("Explained items are those with a parseable reply; failures are "
                 "counted and excluded, never invented.")
    lines.append("")
    lines.append("Selection accounting:")
    lines.append("")
    lines.extend(_markdown_table(
        selection, ["step", "rows_before", "rows_after", "rows_dropped", "reason"]))
    lines.append("")
    return lines


def statistical_lines(out_dir: Path) -> list[str]:
    """Embed the statistical report, or point to it when it is kept separate.

    The narrative stays a standalone qualitative document, so the statistical
    results are not duplicated in full. When they are not embedded the wording
    must still make clear that the analysis exists and is reported elsewhere,
    not that it is missing.
    """
    out_dir = Path(out_dir)
    run_id = out_dir.parent.name
    lines: list[str] = []
    statistical = out_dir.parent / "report" / "report.md"
    if statistical.exists():
        lines.append(statistical.read_text(encoding="utf-8").strip())
    else:
        study_report = PROJECT_ROOT / "reports" / run_id / "study_report.md"
        if study_report.exists():
            lines.append(
                "The statistical results for this study are published in the "
                f"separate per-study report `reports/{run_id}/study_report.md`; "
                "they are not duplicated in this qualitative document. The "
                "statistical tables remain under `results/`; no analysis is "
                "missing.")
        else:
            lines.append(
                "The statistical results for this study are held in the result "
                "tables under `results/` and published by the stage 09 per-study "
                "report (`scripts/09_report.py`); they are not duplicated in this "
                "qualitative document. No analysis is missing.")
    lines.append("")
    return lines


def qualitative_blocks(out_dir: Path, settings: dict) -> dict[str, list[str]]:
    """The reusable qualitative content, without top-level headings."""
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    patterns = _read(tables / "q02_patterns.csv")
    contrast = _read(tables / "q07_reason_contrast.csv")
    examples = _read(tables / "q04_examples.csv")

    recurring: list[str] = []
    recurring.append("A reply can carry up to three reasons, so shares do not sum "
                     "to 100%.")
    recurring.append("")
    for group, title in (("disagreement", "Disagreements"),
                         ("agreement", "Control (agreements)")):
        subset = patterns[patterns["group"] == group]
        recurring.append(f"### {title}")
        recurring.append("")
        if subset.empty:
            recurring.append("No explained rows in this group.")
            recurring.append("")
            continue
        subset = subset.copy()
        subset["share_of_explained"] = subset["share_of_explained"].map(_pct)
        subset["share_of_selected"] = subset["share_of_selected"].map(_pct)
        display, headers = _labelled(
            subset, ["reason", "n", "share_of_explained", "share_of_selected",
                     "top_directions"])
        recurring.extend(_markdown_table(display, headers))
        recurring.append("")
    recurring.append("### Contrast with the control sample")
    recurring.append("")
    display = contrast.copy()
    for column in ("share_disagreements", "share_control",
                   "difference_disagreements_minus_control"):
        display[column] = display[column].map(_pct)
    display, headers = _labelled(
        display, ["reason", "share_disagreements", "share_control",
                  "difference_disagreements_minus_control"])
    recurring.extend(_markdown_table(display, headers))
    recurring.append("")
    recurring.append("A reason common to both groups is not specific to "
                     "disagreements. The control sample is small and its own "
                     "explanations are hypotheses.")
    recurring.append("")

    example_lines: list[str] = []
    example_lines.append("Examples are the highest-confidence hypotheses per reason, "
                         "then the lowest id, shown once each.")
    example_lines.append("")
    disagreement_examples = examples[examples["group"] == "disagreement"]
    order: list[str] = []
    info: dict[str, dict] = {}
    for _, row in disagreement_examples.iterrows():
        identifier = row["id"]
        if identifier not in info:
            info[identifier] = {"row": row, "reasons": []}
            order.append(identifier)
        if row["reason"] not in info[identifier]["reasons"]:
            info[identifier]["reasons"].append(row["reason"])
    if not order:
        example_lines.append("No disagreement examples were available.")
        example_lines.append("")
    for identifier in order[:MAX_EXAMPLES]:
        entry = info[identifier]
        row = entry["row"]
        labels = "; ".join(reason_label_lower(code) for code in entry["reasons"])
        example_lines.append(
            f"- **`{identifier}`** — human `{row['reference']}`, AI "
            f"`{row['predicted']}` (confidence {row['confidence']}; {labels}). "
            f"{row['summary']}")
        if str(row["evidence"]).strip():
            example_lines.append(f"  - evidence: “{row['evidence']}”")
    if len(order) > MAX_EXAMPLES:
        example_lines.append("")
        example_lines.append(f"{len(order) - MAX_EXAMPLES} further disagreements are "
                             "in `qualitative/tables/q04_examples.csv`.")
    example_lines.append("")

    limitations: list[str] = [
        "- The explanations are AI-generated hypotheses; they do not establish "
        "which label is correct.",
        "- Only disagreements are analysed in detail, plus a small seeded control "
        "sample; no base rate is estimated.",
        "- The reason vocabulary is our construct, not a validated taxonomy; "
        "`Other` absorbs what it does not name.",
        "- The explanation model sees the human label, so it may rationalise "
        "toward it; using the same model as the classifier risks self-justification.",
        "- Self-reported confidence ranks examples only, not a probability.",
        "- Where the reference is a majority vote or a single annotator, "
        "`Ambiguous reference label` and `Possible annotation error` cannot be "
        "confirmed.",
        "- Temperature is not controlled and model versions drift.",
    ]

    conclusions: list[str] = []
    top = patterns[patterns["group"] == "disagreement"]
    if not top.empty:
        first = top.iloc[0]
        conclusions.append(
            f"- Across the sampled disagreements the most frequent hypothesis was "
            f"{reason_label_lower(first['reason'])} "
            f"({_pct(first['share_of_explained'])} of explained disagreements).")
    control = patterns[patterns["group"] == "agreement"]
    if not control.empty:
        first_control = control.iloc[0]
        conclusions.append(
            f"- The control sample most often cited "
            f"{reason_label_lower(first_control['reason'])} "
            f"({_pct(first_control['share_of_explained'])}).")
    if len(contrast):
        top_contrast = contrast.iloc[0]
        conclusions.append(
            f"- The reason most concentrated in disagreements was "
            f"{reason_label_lower(top_contrast['reason'])}, "
            f"{_pct(top_contrast['difference_disagreements_minus_control'])} more "
            "common than in the control.")
    conclusions.append("- These are associations within one sample and are not "
                       "causal; a disagreement is not a confirmed model error.")
    conclusions.append("")

    return {"recurring": recurring, "examples": example_lines,
            "limitations": limitations, "conclusions": conclusions}


def build_narrative(out_dir: Path, settings: dict, study_cfg: dict) -> Path:
    out_dir = Path(out_dir)
    report_block = settings.get("report", {})
    title = report_block.get("title") or settings["_meta"]["run_id"]
    author = report_block.get("author") or ""
    run_id = settings["_meta"]["run_id"]
    config_path = settings["_meta"]["study_config_path"]

    lines: list[str] = []
    lines.append(f"# Qualitative analysis of disagreements: {title}")
    lines.append("")
    lines.append(f"Generated by `scripts/08_qualitative.py` from the result and "
                 f"qualitative tables only. {now_iso()}")
    if author:
        lines.append(f"Author: {author}")
    lines.append("")
    lines.append("> **Explanations in this report are AI-generated hypotheses.** "
                 "They are not confirmed causes, not adjudication, and not proof "
                 "that the classifier or the human made a mistake.")
    lines.append("")

    lines.append("## 1. Scope and coverage")
    lines.append("")
    lines.extend(scope_lines(out_dir, settings))

    lines.append("## 2. Statistical results")
    lines.append("")
    lines.extend(statistical_lines(out_dir))

    blocks = qualitative_blocks(out_dir, settings)
    lines.append("## 3. Recurring hypotheses")
    lines.append("")
    lines.extend(blocks["recurring"])
    lines.append("## 4. Illustrative examples (disagreements)")
    lines.append("")
    lines.extend(blocks["examples"])
    lines.append("## 5. Limitations")
    lines.append("")
    lines.extend(blocks["limitations"])
    for limitation in report_block.get("limitations", []) or []:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## 6. Conclusions")
    lines.append("")
    lines.extend(blocks["conclusions"])

    lines.append("## 7. Reproducibility")
    lines.append("")
    lines.append("```")
    portable = _portable(config_path, run_id)
    if portable:
        lines.append(f"uv run python scripts/08_qualitative.py --config {portable} "
                     f"--settings runs/{run_id}/qualitative/settings.yaml --report")
    else:
        lines.append(f"uv run python scripts/08_qualitative.py --config <config> "
                     f"--settings runs/{run_id}/qualitative/settings.yaml --report")
    lines.append(f"uv run python scripts/09_report.py --run-id {run_id}")
    lines.append("```")
    lines.append("")
    lines.append("Seeds, input hashes, prompt templates, and raw replies are recorded "
                 "in the run manifest and `qualitative/attempts.jsonl`. Every number "
                 "here is formatted from a CSV under `results/` or "
                 "`qualitative/tables/`.")
    lines.append("")

    path = out_dir / "narrative_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

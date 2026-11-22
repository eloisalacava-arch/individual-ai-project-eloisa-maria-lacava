"""High-level report builders used by the CLI and the interface.

Each builder composes Markdown from existing tables and, when ReportLab is
available, renders a PDF. No model call and no statistic is recomputed.
"""

from __future__ import annotations

from pathlib import Path

from ..config import load_config
from . import compose, figures, paths, pdf


class ReportBuildError(RuntimeError):
    """Raised when a report cannot be built from the available files."""


def available() -> bool:
    return pdf.pdf_available()


def _figure_payloads(cfg: dict) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for key, builder in (("confusion_matrix", figures.confusion_matrix_png),
                         ("per_class_recall", figures.per_class_recall_png)):
        try:
            payloads[key] = builder(cfg)
        except (FileNotFoundError, ValueError):
            continue
    return payloads


def build_study_pdf(config_path: str | Path, out_dir: str | Path | None = None) -> dict:
    """Write ``study_report.md`` (+ meta) and, if possible, ``study_report.pdf``."""
    markdown_path = compose.write_study_report(config_path, out_dir=out_dir)
    cfg = load_config(config_path)
    run_id = cfg["run"]["run_id"]
    result = {"run_id": run_id, "markdown": markdown_path, "pdf": None}
    if not pdf.pdf_available():
        return result
    target = markdown_path.parent / "study_report.pdf"
    pdf.markdown_to_pdf(
        markdown_path.read_text(encoding="utf-8"),
        target,
        title=cfg.get("report", {}).get("title") or run_id,
        footer=f"run {run_id}",
        hypotheses=compose.qualitative_available(cfg),
        figures=_figure_payloads(cfg),
    )
    result["pdf"] = target
    return result


def build_run_pdf(run_id: str, out_dir: str | Path | None = None) -> dict:
    """Build the per-study report for a run id, resolving its config."""
    config_path = compose.resolve_study_config(run_id)
    if config_path is None:
        raise ReportBuildError(
            f"no study config found for run {run_id!r} (looked in "
            f"runs/{run_id}/study.yaml and configs/*.yaml)")
    return build_study_pdf(config_path, out_dir=out_dir)


def build_qualitative_pdf(study_config_path: str | Path) -> dict:
    """Render the existing qualitative narrative to PDF. No model call."""
    cfg = load_config(study_config_path)
    run_id = cfg["run"]["run_id"]
    source = Path(cfg["_meta"]["output_dir"]) / "qualitative" / "narrative_report.md"
    if not source.exists():
        raise ReportBuildError(
            f"{source} is missing; run stage 08 --report first")
    result: dict = {"run_id": run_id, "markdown": source, "pdf": None}
    if not pdf.pdf_available():
        return result
    target = paths.qualitative_report_path(run_id)
    paths.ensure_dir(target.parent)
    pdf.markdown_to_pdf(source.read_text(encoding="utf-8"), target,
                        title=f"Qualitative analysis — {run_id}", hypotheses=True,
                        footer=f"run {run_id}")
    result["pdf"] = target
    return result


def build_run_reports(run_id: str, out_dir: str | Path | None = None) -> dict:
    """Per-study report plus, when available, the qualitative report PDF."""
    result: dict = {"study": build_run_pdf(run_id, out_dir=out_dir)}
    try:
        config_path = compose.resolve_study_config(run_id)
        result["qualitative"] = build_qualitative_pdf(config_path)
    except ReportBuildError:
        result["qualitative"] = None
    return result


def build_config_reports(config_path: str | Path,
                         out_dir: str | Path | None = None) -> dict:
    result: dict = {"study": build_study_pdf(config_path, out_dir=out_dir)}
    try:
        result["qualitative"] = build_qualitative_pdf(config_path)
    except ReportBuildError:
        result["qualitative"] = None
    return result


def discover_run_ids() -> list[str]:
    root = paths.PROJECT_ROOT / "runs"
    if not root.exists():
        return []
    found = []
    for directory in sorted(root.iterdir()):
        if directory.name.startswith("_") or not directory.is_dir():
            continue
        if (directory / "results" / "t02_accuracy.csv").exists():
            found.append(directory.name)
    return found


def build_comparison_pdf() -> dict:
    """Render the preserved cross-dataset report to PDF, with figures."""
    source = paths.comparison_report_paths()["source"]
    if not source.exists():
        raise ReportBuildError(
            f"{source} is missing; run stage 07 (scripts/07_compare.py) first")
    markdown = source.read_text(encoding="utf-8")
    markdown += (
        "\n## Figures\n\n"
        "<!-- figure: accuracy_kappa | Accuracy and Cohen's kappa with 95% CI -->\n\n"
        "<!-- figure: class_recall | Per-class recall by dataset -->\n\n"
        "<!-- figure: severity | Ordinal distance from the reference label -->\n"
    )
    comparison_dir = paths.PROJECT_ROOT / "comparison"
    figures_payload: dict[str, bytes] = {}
    for key, builder in (("accuracy_kappa", figures.comparison_metrics_png),
                         ("class_recall", figures.comparison_class_recall_png),
                         ("severity", figures.comparison_severity_png)):
        try:
            figures_payload[key] = builder(comparison_dir)
        except FileNotFoundError:
            continue

    paths.ensure_dir(paths.COMPARISON_DIR)
    markdown_path = paths.COMPARISON_DIR / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    result: dict = {"run_id": "comparison", "markdown": markdown_path, "pdf": None}
    if not pdf.pdf_available():
        return result
    target = paths.comparison_report_paths()["pdf"]
    pdf.markdown_to_pdf(markdown, target, title="Cross-dataset comparison",
                        footer="comparison", figures=figures_payload)
    result["pdf"] = target
    return result

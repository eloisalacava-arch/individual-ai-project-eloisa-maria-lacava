"""Report building for the interface.

Wraps the engine's report builders and paths. No model call; every report is
composed from the existing result and qualitative tables.
"""

from __future__ import annotations

from pathlib import Path

from general_agent.report import build as report_build
from general_agent.report import compose as report_compose
from general_agent.report import paths as report_paths
from general_agent.report.build import ReportBuildError


def pdf_available() -> bool:
    return report_build.available()


def config_available(run_id: str) -> bool:
    return report_compose.resolve_study_config(run_id) is not None


def qualitative_included(run_id: str) -> bool:
    config_path = report_compose.resolve_study_config(run_id)
    if config_path is None:
        return False
    try:
        from general_agent.config import load_config
        return report_compose.qualitative_available(load_config(config_path))
    except Exception:
        return False


def study_report_paths(run_id: str) -> dict[str, Path]:
    return report_paths.study_report_paths(run_id)


def qualitative_pdf_path(run_id: str) -> Path:
    return report_paths.qualitative_report_path(run_id)


def comparison_paths() -> dict[str, Path]:
    return report_paths.comparison_report_paths()


def build_study(run_id: str, pdf: bool = True) -> dict:
    """Build the per-study report (and the qualitative PDF when available)."""
    config_path = report_compose.resolve_study_config(run_id)
    if config_path is None:
        raise ReportBuildError(
            f"No study config found for {run_id!r}; cannot build a report.")
    if pdf and pdf_available():
        return report_build.build_run_reports(run_id)
    markdown = report_compose.write_study_report(config_path)
    return {"study": {"markdown": markdown, "pdf": None}, "qualitative": None}


def build_comparison(pdf: bool = True) -> dict:
    if pdf and pdf_available():
        return report_build.build_comparison_pdf()
    source = report_paths.comparison_report_paths()["source"]
    if not source.exists():
        raise ReportBuildError(
            "comparison/report.md is missing; run stage 07 first.")
    target = report_paths.COMPARISON_DIR / "report.md"
    report_paths.ensure_dir(report_paths.COMPARISON_DIR)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return {"markdown": target, "pdf": None}

"""Report paths under the gitignored ``reports/`` directory.

Composed reports are deliberately separate from ``runs/`` and ``comparison/`` so
the shipped case-study artefacts stay byte-identical.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = PROJECT_ROOT / "reports"
COMPARISON_DIR = REPORTS_ROOT / "comparison"
CONFIG_DIR = PROJECT_ROOT / "configs"


def report_dir(run_id: str) -> Path:
    return REPORTS_ROOT / run_id


def study_report_paths(run_id: str) -> dict[str, Path]:
    directory = report_dir(run_id)
    return {
        "dir": directory,
        "markdown": directory / "study_report.md",
        "pdf": directory / "study_report.pdf",
        "meta": directory / "report_meta.json",
    }


def qualitative_report_path(run_id: str) -> Path:
    return report_dir(run_id) / "qualitative_report.pdf"


def comparison_report_paths() -> dict[str, Path]:
    return {
        "dir": COMPARISON_DIR,
        "source": PROJECT_ROOT / "comparison" / "report.md",
        "markdown": COMPARISON_DIR / "report.md",
        "pdf": COMPARISON_DIR / "report.pdf",
        "meta": COMPARISON_DIR / "report_meta.json",
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

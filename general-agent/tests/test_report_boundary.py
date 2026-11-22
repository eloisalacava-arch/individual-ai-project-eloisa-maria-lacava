"""The report package presents; it never recomputes a statistic."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "general_agent"
REPORT = SRC / "report"
FORBIDDEN_MODULES = {"metrics", "intervals"}
FORBIDDEN_SOURCE = ("def confusion_matrix(", "def cohen_kappa", "np.trace",
                    "def wilson", "def bootstrap_core")


def _files() -> list[Path]:
    return sorted(REPORT.glob("*.py"))


def test_report_never_imports_the_metric_or_interval_code():
    offenders = {}
    for path in _files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                tail = node.module.split(".")[-1]
                if tail in FORBIDDEN_MODULES:
                    offenders.setdefault(path.name, set()).add(node.module)
    assert not offenders, f"report package imports evaluation internals: {offenders}"


def test_report_never_defines_a_metric():
    offenders = {}
    for path in _files():
        text = path.read_text(encoding="utf-8")
        hits = [token for token in FORBIDDEN_SOURCE if token in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"report package defines evaluation internals: {offenders}"

"""The interface must call the engine, not reimplement evaluation."""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

FORBIDDEN_MODULES = {"metrics", "intervals", "freeze", "join"}
FORBIDDEN_SOURCE = ("def confusion_matrix", "def cohen_kappa", "np.trace",
                    "def wilson", "def bootstrap_core")


def _app_files() -> list[Path]:
    return sorted(path for path in APP.rglob("*.py"))


def test_no_app_module_imports_the_metric_or_interval_code():
    offenders = {}
    for path in _app_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                tail = node.module.split(".")[-1]
                if tail in FORBIDDEN_MODULES:
                    offenders.setdefault(path.name, set()).add(node.module)
    assert not offenders, f"interface imports evaluation internals: {offenders}"


def test_no_app_module_defines_a_metric():
    offenders = {}
    for path in _app_files():
        text = path.read_text(encoding="utf-8")
        hits = [token for token in FORBIDDEN_SOURCE if token in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"interface defines evaluation internals: {offenders}"

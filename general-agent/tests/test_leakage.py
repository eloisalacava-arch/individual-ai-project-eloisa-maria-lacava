"""Leakage controls: poison marker, import boundary, and file-access scope.

The poison test is the strongest one: a unique marker exists only in the
benchmark's metadata column. If it appears in any prompt, prediction file, log,
or manifest, the separation has failed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from general_agent.blind import run_blind
from general_agent.classify import approve_pilot, run
from general_agent.config import load_config
from helpers import prepare_case
from stubs import StubProvider

MARKER = "LEAK-MARKER-9f3a2c"
SRC = Path(__file__).resolve().parents[1] / "src" / "general_agent"

CLASSIFIER_MODULES = [
    SRC / "classify.py",
    SRC / "prompts.py",
    SRC / "parsing.py",
    SRC / "sealing.py",
    SRC / "providers" / "__init__.py",
    SRC / "providers" / "base.py",
    SRC / "providers" / "opencode_cli.py",
    SRC / "providers" / "openai_compat.py",
]
FORBIDDEN_IMPORTS = {
    "audit", "discover", "blind", "report", "metrics", "evaluate", "join",
    "analyze", "sealed",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module.split(".")[-1] if module else "")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[-1])
    return names


def test_classifier_modules_do_not_import_benchmark_code():
    offenders = {
        path.name: sorted(_imports(path) & FORBIDDEN_IMPORTS)
        for path in CLASSIFIER_MODULES
        if _imports(path) & FORBIDDEN_IMPORTS
    }
    assert not offenders, f"classifier code imports benchmark-side modules: {offenders}"


def test_poison_marker_never_leaves_the_sealed_file(tmp_path):
    config_path, _, _ = prepare_case(
        tmp_path, n=10, sample_n=10, marker=MARKER,
        subgroup_columns=["human_notes"],
    )
    assert run_blind(config_path) == 0

    run_dir = tmp_path / "runs" / "test_run"
    sealed = pd.read_csv(run_dir / "sealed" / "benchmark.csv")
    assert MARKER in set(sealed["human_notes"])  # the marker does exist

    stub = StubProvider(load_config(config_path))
    assert run(config_path, mode="pilot", provider=stub) == 0
    assert approve_pilot(config_path) == 0
    assert run(config_path, mode="full", provider=stub) == 0

    for prompt in stub.prompts:
        assert MARKER not in prompt

    leaked = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if "sealed" in path.parts:
            continue
        if path.suffix in {".csv", ".json", ".jsonl", ".txt"}:
            if MARKER in path.read_text(encoding="utf-8", errors="replace"):
                leaked.append(str(path.relative_to(run_dir)))
    assert not leaked, f"marker leaked into: {leaked}"


def test_classifier_never_opens_benchmark_or_sealed_files(tmp_path, monkeypatch):
    import general_agent.classify as classify_module

    opened: list[str] = []
    original = classify_module.read_table

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(classify_module, "read_table", spy)

    config_path, _, _ = prepare_case(tmp_path, n=6, sample_n=6)
    run_blind(config_path)
    run(config_path, mode="pilot", provider=StubProvider(load_config(config_path)))
    approve_pilot(config_path)
    run(config_path, mode="full", provider=StubProvider(load_config(config_path)))

    assert all("benchmark" not in path and "sealed" not in path for path in opened), opened


def test_blind_input_contains_no_benchmark_column(tmp_path):
    config_path, _, _ = prepare_case(tmp_path, n=8, sample_n=8)
    run_blind(config_path)
    frame = pd.read_csv(tmp_path / "runs" / "test_run" / "blind" / "blind.csv")
    assert list(frame.columns) == ["id", "text"]
    assert frame["text"].str.contains("positive|negative", regex=True).sum() == 0
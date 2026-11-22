"""The qualitative stage is evaluation-side and stays inside its own directory."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from general_agent.qualitative import prompts
from general_agent.qualitative.settings import output_dir
from qualitative_helpers import evaluated_run, run_qualitative

SRC = Path(__file__).resolve().parents[1] / "src" / "general_agent"
CLASSIFIER_MODULES = [
    SRC / "classify.py", SRC / "prompts.py", SRC / "parsing.py",
    SRC / "sealing.py", SRC / "providers" / "base.py",
    SRC / "providers" / "opencode_cli.py", SRC / "providers" / "openai_compat.py",
]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[-1])
    return names


def test_classifier_never_imports_the_qualitative_package():
    for path in CLASSIFIER_MODULES:
        assert "qualitative" not in _imports(path), path.name


def test_one_prompt_carries_one_row_only():
    settings = {"vocabulary_version": 1}
    classes = [{"name": "positive", "definition": "favourable"},
               {"name": "negative", "definition": "unfavourable"}]
    first = "FIRST-ROW-TEXT-alpha"
    second = "SECOND-ROW-TEXT-beta"
    prompt = prompts.build_prompt(settings, classes, {
        "group": "disagreement", "reference": "positive", "predicted": "negative",
        "text": first})
    assert second not in prompt
    assert first in prompt


def _hash_tree(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_qualitative_run_does_not_touch_results_or_predictions(tmp_path):
    study = evaluated_run(tmp_path)
    run_dir = study["run_dir"]
    before = {
        "results": _hash_tree(run_dir / "results"),
        "predictions": hashlib.sha256(
            (run_dir / "predictions.csv").read_bytes()).hexdigest(),
    }
    run_qualitative(study, tmp_path, sample={"n": 4, "control_n": 2},
                    pilot={"disagreements": 2, "agreements": 1},
                    budget={"max_calls": 15})
    after = {
        "results": _hash_tree(run_dir / "results"),
        "predictions": hashlib.sha256(
            (run_dir / "predictions.csv").read_bytes()).hexdigest(),
    }
    assert before == after
    assert not (run_dir / "results" / "q01_explanations.csv").exists()

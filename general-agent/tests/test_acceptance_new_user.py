"""End-to-end acceptance test: a first-time user with two new synthetic CSVs.

The Streamlit file uploader and the live pilot/classify buttons cannot be driven
headlessly (uploads are not settable, and clicking a model button would make a
live call). This test therefore drives every UI step that makes no model call
through Streamlit's harness, and drives the model stages through the same
``app.engine_adapter`` functions the buttons call, using StubProvider. See
``tests/README`` notes in the docstring: file upload and live classification need
manual verification.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import engine_adapter, qualitative_support, registry, workspace  # noqa: E402
from general_agent.config import load_config  # noqa: E402
from general_agent.qualitative.settings import load_settings  # noqa: E402
from general_agent.report import paths as report_paths  # noqa: E402
from helpers import write_rows  # noqa: E402
from stubs import JsonStubProvider, StubProvider  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
RUN_ID = "acceptance_synth"
POSITIVE = ("The unit behaved exactly as described and the setup was quick.",)
NEGATIVE = ("It failed on the second day and support never replied.",)


def _dataset_bytes() -> bytes:
    path = Path("/tmp/_acceptance_dataset.csv")
    rows = []
    for index in range(24):
        text = (POSITIVE[0] if index % 2 == 0 else NEGATIVE[0]) + f" (case {index})"
        rows.append([f"AC-{index + 1:03d}", text, "synthetic"])
    write_rows(path, ["doc_id", "comment", "source"], rows)
    # IDs and text are deliberately unlike the shipped case studies.
    return path.read_bytes()


def _benchmark_bytes() -> bytes:
    path = Path("/tmp/_acceptance_benchmark.csv")
    rows = [[f"AC-{index + 1:03d}", "1" if index % 2 == 0 else "0"]
            for index in range(24)]
    write_rows(path, ["record_id", "verdict"], rows)
    return path.read_bytes()


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _select(app, key, value):
    # Persistent widgets render under a private key and mirror into ``key``.
    widget = next(box for box in app.selectbox if box.key == f"__w_{key}")
    widget.select(value)


def _hash_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_first_time_user_completes_the_whole_workflow(tmp_path, monkeypatch):
    # Two synthetic uploads, staged exactly as the uploader does.
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    monkeypatch.setattr(report_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_paths, "REPORTS_ROOT", tmp_path / "reports")

    project = Path(__file__).resolve().parents[1]
    before_imdb = _hash_tree(project / "runs" / "imdb_case_study" / "results")

    dataset = workspace.stage_upload("s", "dataset", "observations.csv",
                                     _dataset_bytes())
    benchmark = workspace.stage_upload("s", "benchmark", "labels.csv",
                                       _benchmark_bytes())
    dataset_meta = workspace.inspect_table(dataset["path"])
    benchmark_meta = workspace.inspect_table(benchmark["path"])
    assert dataset_meta["columns"] == ["doc_id", "comment", "source"]
    assert benchmark_meta["columns"] == ["record_id", "verdict"]

    def render(**state):
        app = AppTest.from_file(str(APP), default_timeout=120)
        app.session_state["flow"] = "advanced"
        for key, value in state.items():
            app.session_state[key] = value
        app.run()
        return app

    # --- Step 1-2: data plus column mapping -------------------------------
    app = render(step=1, dataset_upload=dataset, benchmark_upload=benchmark,
                 dataset_meta=dataset_meta, benchmark_meta=benchmark_meta)
    assert not app.exception and not app.error
    # Type the study name into the widget, exactly as a user would, then navigate
    # away; the name must survive to the run-creation step.
    next(field for field in app.text_input
         if field.key == "__w_study_name").set_value("acceptance_synth")
    app.run()
    assert app.session_state["study_name"] == "acceptance_synth"
    _button(app, "Continue to columns").click()
    app.run()
    assert not app.exception and not app.error

    _select(app, "dataset_id_column", "doc_id")
    _select(app, "text_column", "comment")
    _select(app, "benchmark_id_column", "record_id")
    _select(app, "label_column", "verdict")
    app.run()
    assert not app.error
    texts = [element.value for element in app.success] + \
        [element.value for element in app.markdown]
    assert any("24 IDs match" in text for text in texts)
    _button(app, "Continue to categories").click()
    app.run()

    # --- Step 3: categories and label mapping ----------------------------
    app.session_state["classes"] = [
        {"name": "positive", "definition": "a favourable overall stance"},
        {"name": "negative", "definition": "an unfavourable overall stance"},
    ]
    app.run()
    mapping_widgets = [s for s in app.selectbox if s.key.startswith("map_")]
    assert len(mapping_widgets) == 2, "expected one mapping widget per raw label value"
    for box in mapping_widgets:
        box.select("positive" if "'1'" in box.label else "negative")
    app.run()
    assert not app.error
    assert app.session_state["label_map"] == {"1": "positive", "0": "negative"}
    _button(app, "Continue to model and limits").click()
    app.run()

    # --- Step 4: model and limits ----------------------------------------
    next(w for w in app.number_input if w.key == "__w_max_calls").set_value(60)
    next(c for c in app.checkbox if c.key == "__w_consent").check()
    app.run()
    assert not app.error
    _button(app, "Continue to validation").click()
    app.run()

    # --- Step 5: validate and seal (no model calls) ----------------------
    _button(app, "Create study and validate").click()
    app.run()
    assert not app.exception and not app.error
    run_dir = tmp_path / "runs" / RUN_ID
    assert (run_dir / "study.yaml").exists()
    decisions = workspace.read_decisions(RUN_ID)
    assert decisions["status"] == "resolved"
    assert workspace.load_manifest(RUN_ID)["checks"]["status"] == "PASS"
    # We never set an explicit ID column on the mapping guess: verify it resolved.
    assert decisions["resolved_mapping"]["id_column"] == "doc_id"
    assert decisions["resolved_mapping"]["label_column"] == "verdict"

    _button(app, "Seal benchmark and continue").click()
    app.run()
    assert not app.exception and not app.error
    flags = workspace.stage_flags(RUN_ID)
    assert flags["sealed"] is True
    blind = pd.read_csv(run_dir / "blind" / "blind.csv", dtype="string")
    assert list(blind.columns) == ["id", "text"]
    sealed = pd.read_csv(run_dir / "sealed" / "benchmark.csv", dtype="string")
    assert set(sealed["id"]) == set(blind["id"])
    # Benchmark is hidden: columns and categories steps are now locked.
    state_app = render(run_id=RUN_ID, step=2)
    disabled = {b.label: b.disabled for b in state_app.button}
    assert next(v for k, v in disabled.items() if k.startswith("2.")) is True

    # --- Blind classification, evaluation, qualitative (stub provider) ---
    config_path = run_dir / "study.yaml"
    stub = StubProvider(load_config(config_path), label="positive")
    assert engine_adapter.pilot(config_path, provider=stub).ok
    assert engine_adapter.approve(config_path).ok
    assert engine_adapter.classify_batch(config_path, None, provider=stub).ok
    assert engine_adapter.evaluate(config_path).code in (0, 1)
    assert engine_adapter.disagreements(config_path).code in (0, 1)

    provider_entry = registry.enabled_providers()[0]
    model_entry = provider_entry["models"][0]
    sidecar = qualitative_support.build_sidecar(provider_entry, model_entry,
                                                sample_n=5, control_n=3,
                                                seed=20261116, max_calls=20,
                                                max_usd=0.10)
    sidecar_path = qualitative_support.write_sidecar(run_dir, sidecar)
    settings = load_settings(config_path, sidecar_path)
    json_stub = JsonStubProvider(settings)
    assert engine_adapter.qualitative_pilot(config_path, sidecar_path,
                                            provider=json_stub).ok
    assert engine_adapter.qualitative_approve(config_path, sidecar_path).ok
    assert engine_adapter.qualitative_run(config_path, sidecar_path,
                                          provider=json_stub).ok
    assert engine_adapter.qualitative_report(config_path, sidecar_path).ok

    explanations = pd.read_csv(run_dir / "qualitative" / "explanations.csv",
                               dtype="string")
    assert len(explanations) == 8
    assert (explanations["status"] == "ok").all()

    # --- Step 7: results, full report, downloads -------------------------
    app = render(step=7, run_id=RUN_ID)
    assert not app.exception and not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Rows analysed"] == "24"
    assert any("Generate full report" in b.label for b in app.button)
    _button(app, "Regenerate full report" if any(
        "Regenerate" in b.label for b in app.button) else "Generate full report").click()
    app.run()
    assert not app.error
    downloads = [d.label for d in app.get("download_button")]
    assert any("Download Full Report (PDF)" in label for label in downloads)
    assert any("Full report (Markdown)" in label for label in downloads)
    report_pdf = tmp_path / "reports" / RUN_ID / "study_report.pdf"
    assert report_pdf.exists() and report_pdf.read_bytes().startswith(b"%PDF-")

    # --- No reuse of case-study data -------------------------------------
    assert RUN_ID not in {"imdb_case_study", "phrasebank_75agree"}
    new_ids = set(blind["id"])
    assert new_ids and all(identifier.startswith("AC-") for identifier in new_ids)
    for case in ("imdb_case_study", "phrasebank_75agree"):
        case_blind = project / "runs" / case / "blind" / "blind.csv"
        if case_blind.exists():
            case_ids = set(pd.read_csv(case_blind, dtype="string")["id"])
            assert not (new_ids & case_ids)
    predictions = pd.read_csv(run_dir / "predictions.csv", dtype="string")
    assert set(predictions["id"]) == new_ids
    assert all(row.startswith("AC-") for row in predictions["id"])

    # Case-study results are untouched by this acceptance run.
    after_imdb = _hash_tree(project / "runs" / "imdb_case_study" / "results")
    assert before_imdb == after_imdb


def test_a_typed_study_name_survives_navigation(tmp_path, monkeypatch):
    """The name in step 1 must still be set when the run is created in step 5."""
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    dataset = workspace.stage_upload("s", "dataset", "o.csv", _dataset_bytes())
    benchmark = workspace.stage_upload("s", "benchmark", "l.csv", _benchmark_bytes())
    dataset_meta = workspace.inspect_table(dataset["path"])
    benchmark_meta = workspace.inspect_table(benchmark["path"])

    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["flow"] = "advanced"
    for key, value in (("dataset_upload", dataset), ("benchmark_upload", benchmark),
                       ("dataset_meta", dataset_meta),
                       ("benchmark_meta", benchmark_meta)):
        app.session_state[key] = value
    app.run()
    next(field for field in app.text_input
         if field.key == "__w_study_name").set_value("manual_acceptance")
    app.run()
    _button(app, "Continue to columns").click()
    app.run()
    # Leaving step 1 must not clear the name.
    assert app.session_state.get("study_name") == "manual_acceptance"


def test_new_user_needs_no_case_study_config(tmp_path, monkeypatch):
    """The generated study config points only at the new uploads."""
    import yaml
    monkeypatch.setattr(workspace, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workspace, "DRAFTS_ROOT", tmp_path / "drafts")
    run_dir = tmp_path / "runs" / RUN_ID
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "inputs" / "dataset.csv").write_bytes(_dataset_bytes())
    (run_dir / "inputs" / "benchmark.csv").write_bytes(_benchmark_bytes())
    from app.config_builder import build_config, write_config

    spec = {
        "run_id": RUN_ID,
        "dataset_rel": "inputs/dataset.csv", "dataset_format": "csv",
        "dataset_delimiter": ",", "dataset_encoding": "utf-8",
        "dataset_id_column": "doc_id", "text_column": "comment",
        "benchmark_rel": "inputs/benchmark.csv", "benchmark_format": "csv",
        "benchmark_delimiter": ",", "benchmark_encoding": "utf-8",
        "benchmark_id_column": "record_id", "label_column": "verdict",
        "label_map": {"1": "positive", "0": "negative"},
        "on_unknown_label": "error", "on_id_mismatch": "error",
        "classes": [{"name": "positive", "definition": "favourable"},
                    {"name": "negative", "definition": "unfavourable"}],
        "ordered": False, "rows_mode": "all", "sample_n": None, "seed": 20261116,
        "task_role": "a classifier", "decision_rule": "",
        "model": {"provider": "opencode_cli", "name": "test/model"},
        "max_calls": 60, "max_usd": 0.1, "price_per_million": {"input": 0.0, "output": 0.0},
        "report_title": RUN_ID, "report_author": "", "task_description": "",
        "reference_description": "",
    }
    config_path = write_config(run_dir, build_config(spec))
    cfg = load_config(config_path)
    assert "imdb" not in str(cfg["dataset"]["path"])
    assert "financial" not in str(cfg["dataset"]["path"])
    assert Path(cfg["dataset"]["path"]).parent == run_dir / "inputs"
    assert Path(cfg["benchmark"]["path"]).parent == run_dir / "inputs"

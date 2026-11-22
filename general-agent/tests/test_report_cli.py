"""The stage 09 CLI composes reports without a model call."""

from __future__ import annotations

import pytest

from general_agent.cli import build_parser, main
from qualitative_helpers import evaluated_run


def test_parser_recognises_the_report_command():
    args = build_parser().parse_args(["report", "--config", "study.yaml"])
    assert args.config == "study.yaml"
    args = build_parser().parse_args(["report", "--comparison", "--no-pdf"])
    assert args.comparison is True and args.no_pdf is True


def test_cli_writes_markdown_only_with_no_pdf(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=12)
    out = tmp_path / "reports"
    code = main(["report", "--config", str(study["config_path"]),
                 "--out", str(out), "--no-pdf"])
    assert code == 0
    assert (out / "study_report.md").exists()
    assert not (out / "study_report.pdf").exists()


def test_cli_writes_a_pdf_when_reportlab_is_present(tmp_path):
    pytest.importorskip("reportlab")
    study = evaluated_run(tmp_path, classes=3, n=12)
    out = tmp_path / "reports"
    code = main(["report", "--config", str(study["config_path"]), "--out", str(out)])
    assert code == 0
    pdf = out / "study_report.pdf"
    assert pdf.exists()
    assert pdf.read_bytes().startswith(b"%PDF-")


def test_cli_report_by_run_id(tmp_path, monkeypatch):
    from general_agent.report import paths
    study = evaluated_run(tmp_path, classes=3, n=12)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    out = tmp_path / "reports"
    code = main(["report", "--run-id", study["run_dir"].name, "--out", str(out)])
    assert code == 0
    assert (out / "study_report.md").exists()


def test_cli_requires_a_target(capsys):
    assert main(["report"]) == 1
    assert "nothing to do" in capsys.readouterr().out


def test_cli_comparison_from_preserved_report(tmp_path, monkeypatch):
    from general_agent.report import paths
    source = paths.PROJECT_ROOT / "comparison" / "report.md"
    if not source.exists():
        pytest.skip("the comparison report is not present in this checkout")
    monkeypatch.setattr(paths, "COMPARISON_DIR", tmp_path / "comparison")
    code = main(["report", "--comparison", "--no-pdf"])
    assert code == 0
    assert (tmp_path / "comparison" / "report.md").exists()

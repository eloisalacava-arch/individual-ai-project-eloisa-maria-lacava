"""The qualitative CLI subcommand parses and can run the no-call report step."""

from __future__ import annotations

from general_agent.cli import build_parser, main
from qualitative_helpers import evaluated_run, run_qualitative


def test_parser_recognises_the_qualitative_command():
    parser = build_parser()
    args = parser.parse_args(["qualitative", "--config", "study.yaml", "--report"])
    assert args.qual_report is True
    assert args.config == "study.yaml"
    args = parser.parse_args(["qualitative", "--config", "s.yaml", "--pilot"])
    assert args.pilot is True and args.qual_report is False
    args = parser.parse_args(["qualitative", "--config", "s.yaml", "--run"])
    assert args.run is True and args.pilot is False


def test_cli_report_makes_no_model_call(tmp_path):
    study = evaluated_run(tmp_path)
    sidecar, _settings = run_qualitative(
        study, tmp_path, sample={"n": 4, "control_n": 2},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 15})
    code = main(["qualitative", "--config", str(study["config_path"]),
                 "--settings", str(sidecar), "--report"])
    assert code == 0
    assert (study["run_dir"] / "qualitative" / "narrative_report.md").exists()

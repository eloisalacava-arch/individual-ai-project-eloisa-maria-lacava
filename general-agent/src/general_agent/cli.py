"""Command-line entry point.

Phase 0-1 exposes two commands:

  general-agent configure --config CONFIG   validate the config, resolve paths,
                                            record defaults and file presence
  general-agent audit     --config CONFIG   full dataset/benchmark audit

Later phases add pilot, classify, evaluate, and report. Keeping the command
surface small means a reader can tell exactly which stage produced a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .audit import run_audit
from .blind import BlindError, run_blind
from .classify import ClassificationError, approve_pilot
from .classify import run as run_classification
from .config import ConfigError, applied_defaults, load_config
from .evaluate.analysis import run_evaluation
from .evaluate.disagreements import run_disagreements
from .io import file_fingerprint, write_json_atomic
from .provenance import load_manifest, now_iso
from .providers.base import ProviderError


def _print_path(label: str, path: str) -> bool:
    fingerprint = file_fingerprint(path)
    if fingerprint["exists"]:
        print(f"  [ok]   {label}: {path} ({fingerprint['bytes']} bytes, "
              f"sha256 {fingerprint['sha256'][:16]}...)")
        return True
    print(f"  [MISSING] {label}: {path}")
    return False


def cmd_configure(args) -> int:
    cfg = load_config(args.config)
    print(f"general-agent {__version__} - configure")
    print(f"config: {cfg['_meta']['config_path']}")
    print(f"config sha256: {cfg['_meta']['config_sha256']}")
    print(f"run_id: {cfg['run']['run_id']}  seed: {cfg['run']['seed']}")
    print(f"provider: {cfg['model']['provider']}  model: {cfg['model']['name']}")
    print(f"classes: {[entry['name'] for entry in cfg['labels']['classes']]}")
    print(f"references: primary='{cfg['benchmark']['label_column']}' "
          f"extras={cfg['benchmark'].get('extra_label_columns', [])}")
    temperature = cfg["model"].get("temperature")
    print(f"temperature: {temperature!r} (requested only; support not guaranteed)")

    print("paths (resolved relative to the config file):")
    all_present = True
    all_present &= _print_path("dataset", cfg["dataset"]["path"])
    all_present &= _print_path("benchmark", cfg["benchmark"]["path"])
    predictions = cfg["dataset"].get("predictions")
    if predictions:
        all_present &= _print_path("predictions", predictions["path"])
        print(f"  reuse={predictions.get('reuse')}: "
              f"{'no model calls for this dataset' if predictions.get('reuse') else 'live classification required'}")
    if cfg["model"]["provider"] == "openai_compatible":
        key_env = cfg["model"].get("api_key_env")
        present = cfg["model"].get("_api_key_env_present")
        print(f"  api key env: {key_env} -> "
              f"{'set' if present else 'NOT SET (required before classify)'}")

    print("defaults applied (keys absent from the config):")
    for section, values in applied_defaults(cfg["_meta"]["config_path"]).items():
        print(f"  {section}: {values}")
    if not applied_defaults(cfg["_meta"]["config_path"]):
        print("  none - every setting is stated explicitly")

    out_dir = Path(cfg["_meta"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(out_dir)
    manifest = {
        **existing,
        "run_id": cfg["run"]["run_id"],
        "last_stage": "00_configure",
        "phase": existing.get("phase") or "00_configure",
        "tool": {"name": "general-agent", "version": __version__},
        "config": {"path": cfg["_meta"]["config_path"],
                   "sha256": cfg["_meta"]["config_sha256"]},
        "seed": cfg["run"]["seed"],
        "model": {"provider": cfg["model"]["provider"],
                  "name": cfg["model"]["name"],
                  "temperature_requested": temperature,
                  "temperature_status": "not_requested" if temperature is None
                  else "requested_unverified"},
        "paths": {
            "dataset": file_fingerprint(cfg["dataset"]["path"]),
            "benchmark": file_fingerprint(cfg["benchmark"]["path"]),
            "predictions": file_fingerprint(predictions["path"]) if predictions else None,
        },
        "all_inputs_present": bool(all_present),
    }
    manifest["stages"] = {
        **existing.get("stages", {}),
        "00_configure": {
            "finished_at": now_iso(),
            "all_inputs_present": bool(all_present),
            "config_sha256": cfg["_meta"]["config_sha256"],
        },
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    print(f"manifest written: {out_dir / 'manifest.json'}")

    if args.print_config:
        effective = {key: value for key, value in cfg.items() if key != "_meta"}
        print(json.dumps(effective, indent=2, default=str))

    if not all_present:
        print("RESULT: configuration valid, but one or more inputs are missing; "
              "fix the paths before running audit")
        return 1
    print("RESULT: configuration valid and all inputs present")
    return 0


def cmd_audit(args) -> int:
    try:
        return run_audit(args.config, strict=args.strict)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1


def cmd_blind(args) -> int:
    try:
        return run_blind(args.config, force=args.force, unlock_only=args.unlock)
    except (BlindError, ConfigError, FileNotFoundError) as exc:
        print(f"BLIND ERROR: {exc}")
        return 1


def cmd_pilot(args) -> int:
    try:
        if args.approve:
            return approve_pilot(args.config, note=args.note or "")
        return run_classification(args.config, mode="pilot", limit=args.limit,
                                  resume=args.resume)
    except (ClassificationError, ConfigError, ProviderError) as exc:
        print(f"PILOT ERROR: {exc}")
        return 1


def cmd_evaluate(args) -> int:
    try:
        return run_evaluation(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1


def cmd_compare(args) -> int:
    from .evaluate.compare import run_comparison
    try:
        return run_comparison(args.configs, output_dir=args.out)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"COMPARE ERROR: {exc}")
        return 1


def cmd_disagreements(args) -> int:
    try:
        return run_disagreements(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1


def cmd_qualitative(args) -> int:
    from .qualitative.run import approve as qualitative_approve
    from .qualitative.run import report as qualitative_report
    from .qualitative.run import run as qualitative_run
    from .qualitative.settings import QualitativeConfigError, QualitativeError
    try:
        if args.qual_report:
            return qualitative_report(args.config, args.settings)
        if args.approve:
            return qualitative_approve(args.config, args.settings,
                                       note=args.note or "")
        mode = "pilot" if args.pilot else "full"
        return qualitative_run(args.config, args.settings, mode=mode,
                               limit=args.limit, resume=args.resume)
    except (QualitativeError, QualitativeConfigError, ConfigError,
            ProviderError) as exc:
        print(f"QUALITATIVE ERROR: {exc}")
        return 1


def cmd_report(args) -> int:
    from .report import build as report_build
    from .report.build import ReportBuildError
    from .report import compose as report_compose
    try:
        built = []
        if args.comparison:
            if args.no_pdf:
                source = report_build.paths.comparison_report_paths()["source"]
                target = report_build.paths.COMPARISON_DIR / "report.md"
                report_build.paths.ensure_dir(report_build.paths.COMPARISON_DIR)
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                built.append({"markdown": target, "pdf": None})
            else:
                built.append(report_build.build_comparison_pdf())
        if args.config:
            if args.no_pdf:
                path = report_compose.write_study_report(args.config, out_dir=args.out)
                built.append({"markdown": path, "pdf": None})
            else:
                built.append(report_build.build_study_pdf(args.config, out_dir=args.out))
        if args.run_id:
            built.append(report_build.build_run_reports(args.run_id, out_dir=args.out))
        if args.all:
            for run_id in report_build.discover_run_ids():
                built.append(report_build.build_run_reports(run_id, out_dir=args.out))
        if not (args.comparison or args.config or args.run_id or args.all):
            print("nothing to do: pass --config, --run-id, --all, or --comparison")
            return 1
        for entry in built:
            _print_report_entry(entry)
        return 0
    except (ReportBuildError, ConfigError, ValueError, OSError) as exc:
        print(f"REPORT ERROR: {exc}")
        return 1


def _print_report_entry(entry) -> None:
    if "study" in entry or "qualitative" in entry:
        for name in ("study", "qualitative"):
            if entry.get(name):
                _print_report_entry(entry[name])
        return
    if entry.get("markdown"):
        print(f"report markdown: {entry['markdown']}")
    if entry.get("pdf"):
        print(f"report pdf:      {entry['pdf']}")


def cmd_classify(args) -> int:
    try:
        return run_classification(args.config, mode="full", limit=args.limit,
                                  resume=args.resume)
    except (ClassificationError, ConfigError, ProviderError) as exc:
        print(f"CLASSIFY ERROR: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="general-agent",
        description="General-purpose AI classification and evaluation agent "
                    "(phase 0-1: configure and audit).",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser(
        "configure", help="validate a config and record resolved paths/defaults"
    )
    configure.add_argument("--config", required=True, help="path to a study YAML")
    configure.add_argument("--print-config", action="store_true",
                           help="also print the fully resolved config as JSON")
    configure.set_defaults(func=cmd_configure)

    audit = subparsers.add_parser(
        "audit", help="audit dataset and benchmark, propose or confirm the mapping"
    )
    audit.add_argument("--config", required=True, help="path to a study YAML")
    audit.add_argument("--strict", action="store_true",
                       help="treat warnings as failures")
    audit.set_defaults(func=cmd_audit)

    blind = subparsers.add_parser(
        "blind", help="build the blind input and seal the benchmark (stage 02)"
    )
    blind.add_argument("--config", required=True)
    blind.add_argument("--force", action="store_true",
                       help="replace an existing blind/sealed pair")
    blind.add_argument("--unlock", action="store_true",
                       help="restore permissions on a sealed directory left locked")
    blind.set_defaults(func=cmd_blind)

    pilot = subparsers.add_parser(
        "pilot", help="classify the pilot rows, or approve them (stage 03)"
    )
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--approve", action="store_true",
                       help="record approval of the existing pilot predictions")
    pilot.add_argument("--note", default="", help="approval note")
    pilot.add_argument("--limit", type=int, default=None)
    pilot.add_argument("--resume", action="store_true")
    pilot.set_defaults(func=cmd_pilot)

    classify = subparsers.add_parser(
        "classify", help="full classification run; requires an approved pilot (stage 04)"
    )
    classify.add_argument("--config", required=True)
    classify.add_argument("--limit", type=int, default=None,
                          help="process at most N pending rows in this invocation")
    classify.add_argument("--resume", action="store_true",
                          help="continue an existing predictions file")
    classify.set_defaults(func=cmd_classify)

    evaluate = subparsers.add_parser(
        "evaluate", help="freeze predictions, open the benchmark, compute metrics (stage 05)"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    disag = subparsers.add_parser(
        "disagreements", help="descriptive disagreement patterns (stage 06)"
    )
    disag.add_argument("--config", required=True)
    disag.set_defaults(func=cmd_disagreements)

    qualitative = subparsers.add_parser(
        "qualitative",
        help="qualitative explanation of disagreements and control (stage 08)"
    )
    qualitative.add_argument("--config", required=True, help="path to the study YAML")
    qualitative.add_argument("--settings", default=None,
                             help="optional qualitative sidecar settings YAML")
    qualitative.add_argument("--pilot", action="store_true",
                             help="explain the small pilot set")
    qualitative.add_argument("--run", action="store_true",
                             help="run the full analysis (also the default when "
                                  "no mode flag is given)")
    qualitative.add_argument("--approve", action="store_true",
                             help="approve the pilot for the full run")
    qualitative.add_argument("--note", default="", help="approval note")
    qualitative.add_argument("--report", dest="qual_report", action="store_true",
                             help="aggregate patterns and write the narrative "
                                  "(no model call)")
    qualitative.add_argument("--limit", type=int, default=None)
    qualitative.add_argument("--resume", action="store_true")
    qualitative.set_defaults(func=cmd_qualitative)

    report = subparsers.add_parser(
        "report",
        help="compose per-study and comparison reports, and export PDFs (stage 09)"
    )
    report.add_argument("--config", default=None, help="path to a study YAML")
    report.add_argument("--run-id", default=None,
                        help="a run id; its study config is resolved automatically")
    report.add_argument("--all", action="store_true",
                        help="every run under runs/ that has evaluated results")
    report.add_argument("--comparison", action="store_true",
                        help="the cross-dataset comparison report")
    report.add_argument("--no-pdf", action="store_true",
                        help="write Markdown only, without ReportLab")
    report.add_argument("--out", default=None, help="override the output directory")
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser(
        "compare", help="cross-dataset tables, figures, and report (stage 07)"
    )
    compare.add_argument("--configs", nargs="+", required=True)
    compare.add_argument("--out", default="comparison")
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"MISSING INPUT: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
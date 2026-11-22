"""Thin adapter over the existing engine stages.

Every function calls an engine entry point and captures its stdout. Exit codes
are mapped to a small vocabulary the interface can render. No evaluation,
sampling, or blinding logic lives here.

A provider can be injected for tests; production passes ``None`` so the engine
builds the configured provider. No stage is ever called on import or on a plain
page render.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

from general_agent import cli
from general_agent.audit import run_audit
from general_agent.blind import BlindError, run_blind
from general_agent.classify import ClassificationError, approve_pilot, run
from general_agent.config import ConfigError, load_config
from general_agent.evaluate.analysis import run_evaluation
from general_agent.evaluate.compare import run_comparison
from general_agent.evaluate.disagreements import run_disagreements
from general_agent.providers.base import ProviderError
from general_agent.qualitative.settings import QualitativeConfigError, QualitativeError

KIND_BY_CODE = {0: "ok", 1: "failed", 2: "precondition", 3: "stopped"}


@dataclass
class StageResult:
    stage: str
    code: int
    log: str
    error: str | None = None
    kind: str = "ok"

    @property
    def ok(self) -> bool:
        return self.code == 0


def _capture(stage: str, function) -> StageResult:
    buffer = io.StringIO()
    caught = (ConfigError, ProviderError, ClassificationError, BlindError,
              QualitativeError, QualitativeConfigError,
              FileNotFoundError, ValueError, OSError)
    try:
        with contextlib.redirect_stdout(buffer):
            code = function()
    except caught as exc:
        return StageResult(stage, 1, buffer.getvalue(),
                           f"{type(exc).__name__}: {exc}", "exception")
    kind = KIND_BY_CODE.get(code, "failed")
    return StageResult(stage, code, buffer.getvalue(), None, kind)


def configure(config_path: str | Path) -> StageResult:
    return _capture("configure", lambda: cli.main(
        ["configure", "--config", str(config_path)]))


def audit(config_path: str | Path, strict: bool = False) -> StageResult:
    return _capture("audit", lambda: run_audit(config_path, strict=strict))


def seal(config_path: str | Path, force: bool = False) -> StageResult:
    return _capture("blind", lambda: run_blind(config_path, force=force))


def pilot(config_path: str | Path, provider=None) -> StageResult:
    return _capture("pilot", lambda: run(
        config_path, mode="pilot", provider=provider))


def approve(config_path: str | Path, note: str = "") -> StageResult:
    return _capture("approve", lambda: approve_pilot(config_path, note=note))


def classify_batch(config_path: str | Path, limit: int | None,
                   provider=None) -> StageResult:
    return _capture("classify", lambda: run(
        config_path, mode="full", limit=limit, resume=True, provider=provider))


def evaluate(config_path: str | Path) -> StageResult:
    return _capture("evaluate", lambda: run_evaluation(config_path))


def disagreements(config_path: str | Path) -> StageResult:
    return _capture("disagreements", lambda: run_disagreements(config_path))


def report(config_path: str | Path, output_dir: str | Path) -> StageResult:
    return _capture("report", lambda: run_comparison([config_path],
                                                     output_dir=str(output_dir)))


def approval_status(config_path: str | Path) -> tuple[bool, str]:
    """Ask the engine whether the existing pilot still authorises a full run."""
    from general_agent.classify import check_approval
    cfg = load_config(config_path)
    return check_approval(Path(cfg["_meta"]["output_dir"]), cfg)


def deliverable_report_path(run_id: str) -> Path:
    return Path(__file__).resolve().parents[1] / "runs" / run_id / "report" / "report.md"


# --------------------------------------------------------------------------- #
# Stage 08 - qualitative disagreement analysis                                #
# --------------------------------------------------------------------------- #

def qualitative_pilot(study_config: str | Path, settings_path: str | Path,
                      provider=None) -> StageResult:
    from general_agent.qualitative.run import run as qualitative_run
    return _capture("qualitative_pilot", lambda: qualitative_run(
        study_config, settings_path, mode="pilot", provider=provider))


def qualitative_approve(study_config: str | Path, settings_path: str | Path,
                        note: str = "") -> StageResult:
    from general_agent.qualitative.run import approve as qualitative_approve
    return _capture("qualitative_approve", lambda: qualitative_approve(
        study_config, settings_path, note=note))


def qualitative_run(study_config: str | Path, settings_path: str | Path,
                    limit: int | None = None, provider=None) -> StageResult:
    from general_agent.qualitative.run import run as qualitative_run
    return _capture("qualitative_run", lambda: qualitative_run(
        study_config, settings_path, mode="full", limit=limit, resume=True,
        provider=provider))


def qualitative_report(study_config: str | Path,
                       settings_path: str | Path) -> StageResult:
    from general_agent.qualitative.run import report as qualitative_report
    return _capture("qualitative_report", lambda: qualitative_report(
        study_config, settings_path))


def qualitative_approval_status(study_config: str | Path,
                                settings_path: str | Path) -> tuple[bool, str]:
    from general_agent.qualitative.run import check_approval
    from general_agent.qualitative.settings import load_settings, output_dir
    settings = load_settings(study_config, settings_path)
    return check_approval(output_dir(settings), settings)

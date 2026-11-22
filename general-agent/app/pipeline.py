"""The single-authorized pipeline behind the simple Analyze screen.

Streamlit-free and disk-derived: each :func:`advance` call performs exactly one
unit of work (a stage, or one classification batch) based on what already exists
under ``runs/<run_id>/``. That makes it resumable after a reload and easy to test
with a stub provider.

It only calls :mod:`app.engine_adapter`, so the engine, reports, and case studies
are untouched. No policy lives here beyond: auto-approve a *clean* pilot, run
qualitative only when disagreements exist, and stop when asked.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app import engine_adapter, qualitative_support, report_support, workspace

try:  # POSIX; the project targets macOS, but degrade safely elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

STOP_RE = re.compile(r"STOPPED EARLY:\s*([A-Za-z0-9_]+)")

# One in-process lock per run; Streamlit serves sessions as threads, so the
# thread lock catches same-process overlap and the file lock catches separate
# processes. The OS releases the file lock when the process exits.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@contextmanager
def _run_lock(run_id: str):
    """Yield True when this caller owns the run, False when another does."""
    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(run_id, threading.Lock())
    if not thread_lock.acquire(blocking=False):
        yield False
        return
    handle = None
    try:
        path = workspace.run_dir(run_id) / ".pipeline.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(),
                                 "started_at": workspace.now_iso()}))
        handle.flush()
        yield True
    finally:
        if handle is not None:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()
        thread_lock.release()


@dataclass
class Advance:
    status: str            # working | done | stopped | needs_pilot_review | blocked | error
    stage: str
    log: str = ""
    detail: str = ""
    action: str = ""
    extra: dict = field(default_factory=dict)


def _stop_reason(log: str) -> str:
    match = STOP_RE.search(log or "")
    return match.group(1) if match else ""


def _qualitative_provider(settings: dict, factory):
    return factory(settings) if factory else None


def _qualitative_pending(run_dir: Path) -> int:
    path = run_dir / "qualitative" / "explanations.csv"
    if not path.exists():
        return 0
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    return int((frame["status"] != "ok").sum())


def advance(run_id: str, cfg_path: str | Path, authorization: dict | None, *,
            stop: bool = False, batch_size: int = 25,
            classification_provider=None,
            qualitative_provider_factory=None) -> Advance:
    """One unit of work, guarded so two sessions cannot drive the same run."""
    if stop:
        return Advance("stopped", "stopped", detail="Stopped by you. Nothing is lost.")
    with _run_lock(run_id) as acquired:
        if not acquired:
            return Advance("busy", "locked",
                           detail="Another session is running this study. "
                                  "Wait for it to finish, then retry.")
        return _advance(run_id, cfg_path, authorization,
                        batch_size=batch_size,
                        classification_provider=classification_provider,
                        qualitative_provider_factory=qualitative_provider_factory)


def _advance(run_id: str, cfg_path: str | Path, authorization: dict | None, *,
             batch_size: int = 25,
             classification_provider=None,
             qualitative_provider_factory=None) -> Advance:
    from general_agent.config import load_config
    cfg = load_config(cfg_path)
    run_dir = workspace.run_dir(run_id)
    auth = authorization or {}
    qualitative = auth.get("qualitative") or {}
    batch = int(auth.get("batch_size") or batch_size)

    if auth.get("config_sha256") and \
            auth["config_sha256"] != cfg["_meta"]["config_sha256"]:
        return Advance("blocked", "authorization",
                       detail="The study changed since it was authorized. "
                              "Review and authorize again.")

    flags = workspace.stage_flags(run_id)

    # 0. create provenance, then validate --------------------------------
    if not flags.get("audited"):
        if not (run_dir / "manifest.json").exists():
            engine_adapter.configure(cfg_path)
        result = engine_adapter.audit(cfg_path)
        if result.code == 2:
            return Advance("blocked", "validate", log=result.log,
                           detail="Validation needs a decision.")
        if result.code != 0:
            return Advance("error", "validate", log=result.log,
                           detail="Validation failed.")
        return Advance("working", "validate", log=result.log,
                       action="Validated the data")

    # 1. seal ------------------------------------------------------------
    if not flags.get("sealed"):
        result = engine_adapter.seal(cfg_path)
        if result.code != 0:
            # A concurrent drive may have sealed it, or the seal may simply be
            # locked while another process classifies. Re-read before failing.
            flags = workspace.stage_flags(run_id)
            if flags.get("sealed"):
                return Advance("working", "seal", log=result.log,
                               action="Benchmark was already sealed")
            return Advance("error", "seal", log=result.log,
                           detail="Sealing the benchmark failed.")
        return Advance("working", "seal", log=result.log,
                       action="Sealed the benchmark")

    # 2. pilot -----------------------------------------------------------
    pilot_path = run_dir / "pilot" / "pilot_predictions.csv"
    if not pilot_path.exists():
        result = engine_adapter.pilot(cfg_path, provider=classification_provider)
        if result.code not in (0, 1, 3):
            return Advance("error", "pilot", log=result.log,
                           detail="The pilot failed.")
        if result.code == 3:
            return Advance("stopped", "pilot", log=result.log,
                           detail="The pilot stopped early.")
        return Advance("working", "pilot", log=result.log,
                       action="Ran the five-row pilot")
    pilot = pd.read_csv(pilot_path, dtype="string", keep_default_na=False)
    pilot_ok = int((pilot["status"] == "ok").sum())
    if pilot_ok != len(pilot) and not flags.get("approved") \
            and not auth.get("allow_unclean_pilot"):
        return Advance("needs_pilot_review", "pilot",
                       detail=f"{pilot_ok} of {len(pilot)} pilot rows parsed.",
                       extra={"pilot_rows": pilot[["id", "label", "status"]]
                              .to_dict("records")})
    if not flags.get("approved"):
        note = ("auto-approved by the simple flow; all pilot rows parsed"
                if pilot_ok == len(pilot)
                else "approved by the simple flow with non-ok pilot rows")
        result = engine_adapter.approve(cfg_path, note=note)
        if result.code != 0:
            return Advance("error", "approve", log=result.log,
                           detail="Could not record the pilot approval.")
        return Advance("working", "approved", log=result.log,
                       action="Approved the pilot")

    # 3. classification (one batch per call) ------------------------------
    counts = workspace.predictions_counts(run_id)
    if counts["total"] == 0 or counts["pending"] > 0:
        result = engine_adapter.classify_batch(cfg_path, batch,
                                               provider=classification_provider)
        if result.code == 2:
            return Advance("blocked", "classify", log=result.log,
                           detail="The run is not approved.")
        if result.code == 3:
            reason = _stop_reason(result.log)
            return Advance("stopped", "classify", log=result.log,
                           detail=f"Stopped early ({reason or 'budget or provider'}). "
                                  "Rows are saved; resume when ready.")
        if result.code not in (0, 1):
            return Advance("error", "classify", log=result.log,
                           detail="Classification failed.")
        done = workspace.predictions_counts(run_id)
        return Advance("working", "classify", log=result.log,
                       action=f"Classified {done['ok']} of {done['total']} rows")

    # 4. evaluation --------------------------------------------------------
    if not flags.get("evaluated"):
        result = engine_adapter.evaluate(cfg_path)
        if not (run_dir / "results" / "t02_accuracy.csv").exists():
            return Advance("error", "evaluate", log=result.log,
                           detail="Evaluation did not produce results.")
        return Advance("working", "evaluate", log=result.log,
                       action="Evaluated against the human benchmark")

    # 5. disagreements -----------------------------------------------------
    if not (run_dir / "results" / "t11_disagreement_pairs.csv").exists():
        result = engine_adapter.disagreements(cfg_path)
        if result.code not in (0, 1):
            return Advance("error", "disagreements", log=result.log,
                           detail="Disagreement analysis failed.")
        return Advance("working", "disagreements", log=result.log,
                       action="Summarised disagreements")

    # 6. qualitative, only when there is something to explain --------------
    t08 = run_dir / "results" / "t08_disagreements.csv"
    n_disagreements = int(len(pd.read_csv(t08, dtype="string"))) if t08.exists() else 0
    qualitative_enabled = bool(qualitative.get("enabled"))
    if qualitative_enabled and n_disagreements > 0:
        sidecar = qualitative_support.settings_path(run_dir)
        if not sidecar.exists():
            qualitative_support.write_sidecar(run_dir, qualitative["settings"])
            return Advance("working", "qualitative settings",
                           action="Prepared the qualitative settings")
        pilot = run_dir / "qualitative" / "pilot" / "pilot_explanations.csv"
        if not pilot.exists():
            result = engine_adapter.qualitative_pilot(
                cfg_path, sidecar,
                provider=_qualitative_provider(qualitative["settings"],
                                               qualitative_provider_factory))
            if result.code not in (0, 1, 3):
                return Advance("error", "qualitative pilot", log=result.log,
                               detail="The qualitative pilot failed.")
            if result.code == 3:
                return Advance("stopped", "qualitative pilot", log=result.log,
                               detail="The qualitative pilot stopped early.")
            return Advance("working", "qualitative pilot", log=result.log,
                           action="Ran the three-item qualitative pilot")
        qpilot = pd.read_csv(pilot, dtype="string", keep_default_na=False)
        qok = int((qpilot["status"] == "ok").sum())
        approved = (run_dir / "qualitative" / "approval.json").exists()
        if qok != len(qpilot) and not approved:
            return Advance("needs_pilot_review", "qualitative pilot",
                           detail=f"{qok} of {len(qpilot)} qualitative pilot rows "
                                  "parsed.",
                           extra={"qualitative": True,
                                  "pilot_rows": qpilot[["id", "label", "status"]]
                                  .to_dict("records")})
        if not approved:
            result = engine_adapter.qualitative_approve(
                cfg_path, sidecar, note="auto-approved by the simple flow")
            if result.code != 0:
                return Advance("error", "qualitative approve", log=result.log,
                               detail="Could not approve the qualitative pilot.")
            return Advance("working", "qualitative approved", log=result.log,
                           action="Approved the qualitative pilot")
        if not (run_dir / "qualitative" / "explanations.csv").exists() \
                or _qualitative_pending(run_dir) > 0:
            result = engine_adapter.qualitative_run(
                cfg_path, sidecar, limit=10,
                provider=_qualitative_provider(qualitative["settings"],
                                               qualitative_provider_factory))
            if result.code == 3:
                return Advance("stopped", "qualitative", log=result.log,
                               detail="The qualitative run stopped early.")
            if result.code not in (0, 1):
                return Advance("error", "qualitative", log=result.log,
                               detail="Qualitative explanation failed.")
            return Advance("working", "qualitative", log=result.log,
                           action="Explained disagreements")

    # 7. statistical report (embedded by the qualitative narrative) --------
    if not (run_dir / "report" / "report.md").exists():
        result = engine_adapter.report(cfg_path, run_dir / "report")
        if result.code != 0:
            return Advance("error", "statistical report", log=result.log,
                           detail="Could not build the statistical report.")
        return Advance("working", "statistical report", log=result.log,
                       action="Built the statistical report")

    # 8. qualitative narrative --------------------------------------------
    if qualitative_enabled and n_disagreements > 0:
        narrative = run_dir / "qualitative" / "narrative_report.md"
        if not narrative.exists():
            result = engine_adapter.qualitative_report(
                cfg_path, qualitative_support.settings_path(run_dir))
            if result.code != 0:
                return Advance("error", "qualitative report", log=result.log,
                               detail="Could not build the qualitative narrative.")
            return Advance("working", "qualitative report", log=result.log,
                           action="Built the qualitative narrative")

    # 9. study report + PDF ------------------------------------------------
    if not report_support.study_report_paths(run_id)["markdown"].exists():
        try:
            report_support.build_study(run_id)
        except Exception as exc:  # surfaced to the user, never swallowed
            return Advance("error", "study report",
                           detail=f"{type(exc).__name__}: {exc}")
        return Advance("working", "study report",
                       action="Generated the study report and PDF")

    return Advance("done", "done",
                   detail="The analysis is complete.",
                   action="Finished")

"""Stages 08 - pilot, approval, and the resumable explanation run.

Mirrors the classification runner's discipline: one JSONL record per attempt,
atomic saves, a call cap counted from the logs, and an approval file that binds
the model and both prompt-template hashes. It reads the frozen predictions and
the sealed reference; it never writes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..io import write_json_atomic
from ..provenance import (
    append_jsonl, count_jsonl, file_sha256, now_iso, sha256_text, update_manifest,
)
from ..usage import extract_step_usage
from . import prompts
from .schema import CURRENT_VOCABULARY_VERSION, parse_explanation, reason_codes
from .select import (
    build_selection, load_selection, require_frozen, selection_signature,
    stored_signature, write_selection,
)
from .settings import QualitativeError, load_settings, output_dir

EXPLANATION_COLUMNS = [
    "group", "id", "reference", "predicted", "summary", "reasons", "reasons_raw",
    "evidence", "evidence_valid", "what_would_resolve", "confidence", "status",
    "attempts", "model",
]
RAW_LIMIT = 200_000
TERMINAL_STOP = {"budget_calls_exhausted", "budget_usd_exhausted"}


def _paths(out_dir: Path, mode: str) -> dict:
    if mode == "pilot":
        return {
            "explanations": out_dir / "pilot" / "pilot_explanations.csv",
            "log": out_dir / "pilot" / "pilot_attempts.jsonl",
            "report": out_dir / "pilot" / "pilot_report.json",
        }
    return {
        "explanations": out_dir / "explanations.csv",
        "log": out_dir / "attempts.jsonl",
        "report": None,
    }


def _total_calls(out_dir: Path) -> tuple[int, int]:
    return (count_jsonl(_paths(out_dir, "pilot")["log"]),
            count_jsonl(_paths(out_dir, "full")["log"]))


def _estimate_usd(prompt_chars: int, settings: dict,
                  output_tokens: int = 300) -> float:
    prices = settings["budget"].get("price_per_million")
    if not prices:
        return 0.0
    return (prompt_chars / 4.0 * float(prices["input"])
            + output_tokens * float(prices["output"])) / 1_000_000


def _init_rows(items: pd.DataFrame, model: str) -> list[dict]:
    rows = []
    for _, item in items.iterrows():
        rows.append({
            "group": item["group"], "id": item["id"],
            "reference": item["reference"], "predicted": item["predicted"],
            "summary": "", "reasons": "", "reasons_raw": "", "evidence": "",
            "evidence_valid": "", "what_would_resolve": "", "confidence": "",
            "status": "pending", "attempts": 0, "model": model,
        })
    return rows


def _save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=EXPLANATION_COLUMNS)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _load(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    if list(frame.columns) != EXPLANATION_COLUMNS:
        raise QualitativeError(
            f"{path.name} columns are {list(frame.columns)}, expected "
            f"{EXPLANATION_COLUMNS}")
    frame["attempts"] = pd.to_numeric(frame["attempts"], errors="coerce").fillna(0).astype(int)
    return frame.to_dict(orient="records")


def _pilot_items(selected: pd.DataFrame, settings: dict) -> pd.DataFrame:
    pilot = settings["pilot"]
    parts = []
    for group, key in (("disagreement", "disagreements"),
                       ("agreement", "agreements")):
        subset = selected[selected["group"] == group]
        n = int(pilot.get(key, 0))
        if n > 0:
            parts.append(subset.head(n))
    if not parts:
        raise QualitativeError("the pilot is empty; check pilot settings")
    return pd.concat(parts, ignore_index=True)


def _ensure_selection(out_dir: Path, cfg: dict, settings: dict) -> pd.DataFrame:
    current = selection_signature(settings)
    existing = out_dir / "selected.csv"
    if existing.exists():
        if stored_signature(out_dir) == current:
            return load_selection(out_dir)
        if (out_dir / "explanations.csv").exists():
            raise QualitativeError(
                "the qualitative settings changed after explanations were "
                "written; start a new qualitative run rather than mixing draws")
    selected, accounting = build_selection(out_dir.parent, cfg, settings)
    if selected.empty:
        raise QualitativeError("no disagreements or control rows were selected")
    write_selection(out_dir, selected, accounting, settings)
    return selected


def _seed_from_pilot(rows: list[dict], out_dir: Path, settings: dict) -> int:
    """Reuse already-explained pilot rows in the full run.

    The pilot is a review gate, but its answers come from the same model and the
    same prompt as the full run (the approval binds both), so re-running them
    would only pay twice for the same explanation. Only rows whose pilot status
    is ``ok`` and whose id is in the selection are copied; nothing is imputed.
    Returns the number of rows seeded.
    """
    path = out_dir / "pilot" / "pilot_explanations.csv"
    if not path.exists():
        return 0
    pilot = pd.read_csv(path, dtype="string", keep_default_na=False)
    if "model" in pilot.columns:
        models = {value for value in pilot["model"] if str(value).strip()}
        if models and models != {settings["model"]["name"]}:
            return 0  # a different model; never reuse another model's answers
    ok_rows = pilot[pilot["status"] == "ok"].to_dict(orient="records")
    by_id = {record["id"]: record for record in ok_rows}
    fields = ("summary", "reasons", "reasons_raw", "evidence", "evidence_valid",
              "what_would_resolve", "confidence")
    seeded = 0
    for row in rows:
        record = by_id.get(row["id"])
        if record is None:
            continue
        for field in fields:
            row[field] = record.get(field, "")
        row["status"] = "ok"
        row["attempts"] = int(record.get("attempts") or 1)
        seeded += 1
    return seeded


def _quarantine(out_dir: Path) -> Path:
    path = out_dir / "quarantine"
    path.mkdir(parents=True, exist_ok=True)
    (path / "notice.txt").write_text(
        "This directory is the working directory for explanation calls. It "
        "contains no study data.\n", encoding="utf-8")
    entries = sorted(p.name for p in path.iterdir())
    if entries != ["notice.txt"]:
        raise QualitativeError(
            f"quarantine must contain only notice.txt, found {entries}")
    return path


def check_approval(out_dir: Path, settings: dict) -> tuple[bool, str]:
    path = out_dir / "approval.json"
    if not path.exists():
        return False, (f"{path} is missing: run the pilot (--pilot) and approve it "
                       "before the full run")
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("model") != settings["model"]["name"]:
        return False, (f"approval was for model {approval.get('model')!r}, settings "
                       f"now say {settings['model']['name']!r}")
    if approval.get("provider") != settings["model"]["provider"]:
        return False, "provider changed since approval; re-run the pilot"
    current = prompts.prompt_template_hashes(settings, settings["_meta"]["classes"])
    if approval.get("prompt_template_hashes") != current:
        return False, "prompt template changed since approval; re-run the pilot"
    if int(approval.get("vocabulary_version", -1)) != int(
            settings.get("vocabulary_version", CURRENT_VOCABULARY_VERSION)):
        return False, "the vocabulary changed since approval; re-run the pilot"
    pilot_path = out_dir / "pilot" / "pilot_explanations.csv"
    if pilot_path.exists() and approval.get("pilot_explanations_sha256") != \
            file_sha256(pilot_path):
        return False, "pilot explanations changed since approval; review and re-approve"
    return True, f"approved at {approval.get('approved_at')}"


def approve(study_config_path: str | Path,
            settings_path: str | Path | None = None, note: str = "") -> int:
    settings = load_settings(study_config_path, settings_path)
    out_dir = output_dir(settings)
    pilot_path = out_dir / "pilot" / "pilot_explanations.csv"
    if not pilot_path.exists():
        print(f"ABORT: no pilot explanations at {pilot_path}; run --pilot first")
        return 1
    rows = _load(pilot_path)
    ok = sum(1 for row in rows if row["status"] == "ok")
    approval = {
        "approved_at": now_iso(),
        "note": note,
        "model": settings["model"]["name"],
        "provider": settings["model"]["provider"],
        "prompt_template_hashes": prompts.prompt_template_hashes(
            settings, settings["_meta"]["classes"]),
        "vocabulary_version": int(settings.get("vocabulary_version", 1)),
        "prompt_version": int(settings.get("prompt_version", 1)),
        "n_pilot": len(rows),
        "n_ok": ok,
        "pilot_explanations_sha256": file_sha256(pilot_path),
        "temperature_requested": settings["model"].get("temperature"),
        "temperature_status": "not_requested"
        if settings["model"].get("temperature") is None else "requested_unverified",
        "settings_sha256": settings["_meta"]["settings_sha256"],
        "study_config_sha256": settings["_meta"]["study_config_sha256"],
    }
    write_json_atomic(out_dir / "approval.json", approval)
    print(f"qualitative pilot approved: {out_dir / 'approval.json'}")
    print(f"  model: {approval['model']}")
    print(f"  pilot rows: {len(rows)} ({ok} ok)")
    if ok != len(rows):
        print("WARNING: approved with non-ok pilot rows; review them")
    return 0


def _select_batch(selected: pd.DataFrame, settings: dict, mode: str) -> pd.DataFrame:
    return _pilot_items(selected, settings) if mode == "pilot" else selected


def run(study_config_path: str | Path, settings_path: str | Path | None = None,
        mode: str = "full", limit: int | None = None, resume: bool = False,
        provider=None) -> int:
    if mode not in ("pilot", "full"):
        raise QualitativeError(f"mode must be 'pilot' or 'full', got {mode!r}")
    from ..providers import build_provider

    settings = load_settings(study_config_path, settings_path)
    cfg = load_config(settings["_meta"]["study_config_path"])
    out_dir = output_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    require_frozen(out_dir.parent)
    selected = _ensure_selection(out_dir, cfg, settings)
    items = _select_batch(selected, settings, mode).reset_index(drop=True)
    classes = settings["_meta"]["classes"]
    reasons_allowed = reason_codes(int(settings.get("vocabulary_version", 1)))

    if mode == "full":
        approved, detail = check_approval(out_dir, settings)
        if not approved:
            print(f"ABORT: {detail}")
            return 2
        print(f"approval: {detail}")

    paths = _paths(out_dir, mode)
    items = items.drop(columns=["text"], errors="ignore")
    expected_ids = items["id"].astype(str).tolist()
    rows = None
    if paths["explanations"].exists() and resume:
        rows = _load(paths["explanations"])
        if [row["id"] for row in rows] != expected_ids:
            raise QualitativeError(
                "explanation file IDs do not match the selection; refusing to resume")
        print(f"resume: {sum(1 for r in rows if r['status'] == 'ok')} row(s) already ok")
    elif paths["explanations"].exists() and mode == "full":
        existing = _load(paths["explanations"])
        if any(row["status"] != "pending" for row in existing):
            raise QualitativeError(
                f"{paths['explanations']} already holds results; pass --resume or "
                "remove the file to start over")
    seeded_from_pilot = 0
    if rows is None:
        rows = _init_rows(items, settings["model"]["name"])
        if mode == "full":
            seeded_from_pilot = _seed_from_pilot(rows, out_dir, settings)
            if seeded_from_pilot:
                print(f"seeded {seeded_from_pilot} row(s) from the approved pilot; "
                      "they will not be called again")
    text_by_id = dict(zip(selected["id"], selected["text"]))
    group_by_id = dict(zip(selected["id"], selected["group"]))
    ref_by_id = dict(zip(selected["id"], selected["reference"]))
    pred_by_id = dict(zip(selected["id"], selected["predicted"]))

    provider = provider or build_provider(settings, workdir=str(_quarantine(out_dir)))
    print(f"script: general-agent qualitative ({mode})")
    print(f"started: {now_iso()}")
    print(f"provider: {provider.name}  model: {provider.model}")
    print(f"vocabulary_version: {settings.get('vocabulary_version', 1)}")
    print(f"attempt log: {paths['log']}")
    _save(paths["explanations"], rows)

    max_calls = int(settings["budget"]["max_calls"])
    max_usd = settings["budget"].get("max_usd")
    pilot_calls, full_calls = _total_calls(out_dir)
    calls_used = pilot_calls + full_calls
    estimated_usd = actual_usd = 0.0
    timeout_s = int(settings["model"].get("timeout_s", 180))
    max_attempts = int(settings["model"].get("max_attempts", 2))
    print(f"budget: {calls_used}/{max_calls} calls already used "
          f"(pilot {pilot_calls} + full {full_calls}); max_usd={max_usd}")

    pending = [row for row in rows if row["status"] != "ok"]
    if limit:
        pending = pending[:limit]
    print(f"rows: {len(rows)} total -> {len(pending)} pending in this run "
          f"(reason: status != 'ok'" + (f", limited to {limit}" if limit else "") + ")")

    stopped = None
    for index, row in enumerate(pending, 1):
        if stopped:
            break
        item = {
            "group": group_by_id[row["id"]],
            "reference": ref_by_id[row["id"]],
            "predicted": pred_by_id[row["id"]],
            "text": text_by_id[row["id"]],
        }
        prompt = prompts.build_prompt(settings, classes, item)
        prompts.validate_rendering(prompt, item["text"])
        template_sha = prompts.prompt_template_sha(
            settings, classes, prompts.variant_for(item["group"]))
        outcome = None
        attempts_used = 0
        for attempt in range(1, max_attempts + 1):
            if calls_used >= max_calls:
                stopped = "budget_calls_exhausted"
                break
            projected = max(estimated_usd, actual_usd) + _estimate_usd(len(prompt), settings)
            if max_usd is not None and projected > float(max_usd):
                stopped = "budget_usd_exhausted"
                break

            started = now_iso()
            result = provider.call(prompt, timeout_s)
            calls_used += 1
            attempts_used = attempt
            estimated_usd = projected
            raw = result.raw_stdout or ""
            status, explanation, detail = "ok", None, ""
            if result.status == "ok" and result.text:
                status, explanation, detail = parse_explanation(
                    result.text, reasons_allowed, item["text"])
            else:
                status = result.status
                detail = result.error or ""

            append_jsonl(paths["log"], {
                "ts_start": started,
                "ts_end": now_iso(),
                "run_id": cfg["run"]["run_id"],
                "mode": mode,
                "group": row["group"],
                "id": row["id"],
                "attempt": attempt,
                "provider": provider.name,
                "model": provider.model,
                "prompt_sha256": sha256_text(prompt),
                "prompt_template_sha256": template_sha,
                "vocabulary_version": settings.get("vocabulary_version", 1),
                "temperature_requested": provider.temperature,
                "temperature_status": provider.temperature_status,
                "prompt": prompt,
                "raw_stdout": raw[:RAW_LIMIT],
                "raw_stdout_truncated": len(raw) > RAW_LIMIT,
                "raw_stdout_sha256": sha256_text(raw),
                "raw_stderr": (result.raw_stderr or "")[:4000],
                "returncode": result.returncode,
                "elapsed_s": result.elapsed_s,
                "reply_sha256": sha256_text(result.text) if result.text else None,
                "status": status,
                "detail": detail,
                "parse_method": result.extra.get("parse_method"),
                "usage": result.extra.get("usage"),
                "cost_usd": result.extra.get("cost_usd"),
                "estimated_cost_usd": round(_estimate_usd(len(prompt), settings), 10),
            })

            call_cost = result.extra.get("cost_usd")
            if isinstance(call_cost, (int, float)):
                actual_usd += float(call_cost)

            if status == "ok" and explanation is not None:
                row["summary"] = explanation["summary"]
                row["reasons"] = ";".join(explanation["reasons"])
                row["reasons_raw"] = ";".join(explanation["reasons_raw"])
                row["evidence"] = explanation["evidence"]
                row["evidence_valid"] = bool(explanation["evidence_valid"])
                row["what_would_resolve"] = explanation["what_would_resolve"]
                row["confidence"] = explanation["confidence"]
                row["status"] = "ok"
                row["attempts"] = attempt
                outcome = "ok"
                break
            outcome = status
            if status.startswith("blocked_"):
                stopped = status
                break

        if outcome == "ok":
            pass
        elif stopped in TERMINAL_STOP:
            pass  # row stays pending and resumable
        else:
            row["status"] = outcome or "failed_all_attempts"
            row["attempts"] = attempts_used
        _save(paths["explanations"], rows)
        print(f"[{index}/{len(pending)}] {row['group']} {row['id']}: "
              f"status={row['status']} attempts={row['attempts']} "
              f"calls={calls_used}/{max_calls}")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"status counts: {counts}")
    final_pilot, final_full = _total_calls(out_dir)
    final_calls = final_pilot + final_full
    print(f"calls: {final_calls} used in total (pilot {final_pilot} + full {final_full}); "
          f"actual cost ${actual_usd:.6f}")

    if mode == "pilot":
        report = {
            "created_at": now_iso(),
            "n_rows": len(rows),
            "n_ok": counts.get("ok", 0),
            "statuses": counts,
            "model": provider.model,
            "prompt_template_hashes": prompts.prompt_template_hashes(settings, classes),
            "vocabulary_version": settings.get("vocabulary_version", 1),
            "calls_used": final_calls,
            "actual_usd": round(actual_usd, 8),
            "estimated_usd": round(estimated_usd, 6),
            "next_step": "review pilot_explanations.csv, then approve with "
                         "scripts/08_qualitative.py --approve",
        }
        write_json_atomic(paths["report"], report)
        print(f"pilot report: {paths['report']}")

    update_manifest(out_dir.parent, "08_qualitative", {
        "finished_at": now_iso(),
        "mode": mode,
        "provider": provider.describe(),
        "model": provider.model,
        "prompt_template_hashes": prompts.prompt_template_hashes(settings, classes),
        "vocabulary_version": settings.get("vocabulary_version", 1),
        "explanations_path": str(paths["explanations"]),
        "explanations_sha256": file_sha256(paths["explanations"]),
        "n_rows": len(rows),
        "status_counts": counts,
        "seeded_from_pilot": seeded_from_pilot,
        "calls_used_total": final_calls,
        "calls_pilot": final_pilot,
        "calls_full": final_full,
        "max_calls": max_calls,
        "actual_usd": round(actual_usd, 8),
        "estimated_usd": round(estimated_usd, 6),
        "estimated_usd_note": "rough estimate from characters/4 and 300 output "
                              "tokens; not billing data",
        "sample": settings["sample"],
        "pilot": settings["pilot"],
        "selection_signature": selection_signature(settings),
        "stopped_reason": stopped,
        "hypotheses_only": True,
    })

    if stopped:
        print(f"STOPPED EARLY: {stopped}; remaining rows are pending and resumable")
        return 3
    if counts.get("ok", 0) != len(rows):
        print("RESULT: incomplete; some rows are not ok")
        return 1
    print("RESULT: all selected rows explained")
    return 0


def report(study_config_path: str | Path,
           settings_path: str | Path | None = None) -> int:
    """Aggregate patterns and compose the narrative. No model call."""
    from . import narrative, patterns

    settings = load_settings(study_config_path, settings_path)
    cfg = load_config(settings["_meta"]["study_config_path"])
    out_dir = output_dir(settings)
    if not (out_dir / "explanations.csv").exists():
        print(f"ABORT: {out_dir / 'explanations.csv'} is missing; run the "
              "explanations first")
        return 1
    written = patterns.write_tables(out_dir)
    path = narrative.build_narrative(out_dir, settings, cfg)
    update_manifest(out_dir.parent, "08_qualitative_report", {
        "finished_at": now_iso(),
        "tables": {name: str(value) for name, value in written.items()},
        "narrative_path": str(path),
        "narrative_sha256": file_sha256(path),
        "hypotheses_only": True,
    })
    print(f"qualitative tables: {out_dir / 'tables'}")
    print(f"narrative report: {path}")
    return 0

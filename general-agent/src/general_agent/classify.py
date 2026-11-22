"""Stages 03-04 - pilot and full classification.

Reads only ``blind/blind.csv`` (columns ``id, text``). Never opens the sealed
benchmark, the source dataset, or the source benchmark.

Per attempt it appends one JSONL record containing the full prompt, the full raw
reply (capped and flagged if enormous), model identity, attempt number, timing,
and outcome. Predictions are saved atomically after every row, so an interrupted
run loses at most the in-flight observation.

Budget is enforced in two ways: a hard model-call cap (counted from the attempt
log, so resumes cannot bypass it) and a rough spend estimate. The estimate is a
guardrail, not billing data, and is labelled as such in the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .sealing import lock_sealed, unlock_sealed
from .config import class_names, load_config
from .io import read_table, write_json_atomic
from .parsing import extract_label
from .prompts import build_prompt, prompt_template_sha, validate_rendering
from .provenance import (
    append_jsonl, count_jsonl, file_sha256, now_iso, sha256_text, update_manifest,
)
from .usage import extract_step_usage

PRED_COLUMNS = ["id", "label", "status", "attempts", "model"]
RAW_LIMIT = 200_000
TERMINAL_STOP = {"budget_calls_exhausted", "budget_usd_exhausted"}


class ClassificationError(RuntimeError):
    """Raised when a stage precondition is not met."""


def _paths(out_dir: Path, mode: str) -> dict:
    if mode == "pilot":
        base = out_dir / "pilot"
        return {
            "predictions": base / "pilot_predictions.csv",
            "log": base / "pilot_attempts.jsonl",
            "report": base / "pilot_report.json",
            "approval": out_dir / "pilot" / "approval.json",
        }
    return {
        "predictions": out_dir / "predictions.csv",
        "log": out_dir / "logs" / "attempts.jsonl",
        "report": None,
        "approval": out_dir / "pilot" / "approval.json",
    }


def _total_calls(out_dir: Path) -> tuple[int, int]:
    """Calls already spent for this run_id, split pilot vs full.

    The cap is a total for the run, not per file. Counting per file would make
    the pilot invisible to the full-run budget, so a config saying 300 would
    allow 305 calls.
    """
    return (
        count_jsonl(_paths(out_dir, "pilot")["log"]),
        count_jsonl(_paths(out_dir, "full")["log"]),
    )


def _totals_from_logs(out_dir: Path, cfg: dict) -> tuple[float, float, dict]:
    """Recompute cost and token totals from the attempt logs.

    The logs are the durable record: a resumed invocation makes no calls, so an
    in-memory accumulator would report zero and overwrite the real totals.
    Records written before per-call estimates were logged are re-estimated from
    their prompt length, so old runs keep a coherent estimate.
    """
    actual = 0.0
    estimated = 0.0
    tokens = {"input": 0, "output": 0, "reasoning": 0,
              "cache_read": 0, "cache_write": 0}
    for mode in ("pilot", "full"):
        path = _paths(out_dir, mode)["log"]
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cost = record.get("cost_usd")
            usage = record.get("usage")
            if cost is None or usage is None:
                # Older records predate cost logging; the raw output still holds
                # the step_finish record. Derive from it instead of editing logs.
                derived = extract_step_usage(record.get("raw_stdout") or "")
                if derived:
                    if cost is None:
                        cost = derived["cost_usd"]
                    if usage is None:
                        usage = derived["tokens"]
            if isinstance(cost, (int, float)):
                actual += float(cost)
            estimate = record.get("estimated_cost_usd")
            if not isinstance(estimate, (int, float)):
                estimate = _estimate_usd(len(record.get("prompt") or ""), cfg)
            estimated += float(estimate)
            for key, value in (usage or {}).items():
                if key in tokens and isinstance(value, (int, float)):
                    tokens[key] += int(value)
    return actual, estimated, tokens


def _read_blind(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ClassificationError(
            f"{path} is missing: run stage 02 (blind) before classifying"
        )
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    if list(frame.columns) != ["id", "text"]:
        raise ClassificationError(
            f"blind input must have exactly ['id', 'text'], found {list(frame.columns)}"
        )
    if frame["id"].duplicated().any():
        raise ClassificationError("blind input has duplicate IDs")
    if frame["text"].str.strip().eq("").any():
        raise ClassificationError("blind input has blank observation text")
    return frame


def _init_predictions(path: Path, ids: list[str], model: str) -> list[dict]:
    rows = [{"id": identifier, "label": "", "status": "pending",
             "attempts": 0, "model": model} for identifier in ids]
    _save_predictions(path, rows)
    return rows


def _load_predictions(path: Path) -> list[dict]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    if list(frame.columns) != PRED_COLUMNS:
        raise ClassificationError(
            f"predictions file columns are {list(frame.columns)}, expected {PRED_COLUMNS}"
        )
    frame["attempts"] = pd.to_numeric(frame["attempts"], errors="coerce").fillna(0).astype(int)
    return frame.to_dict(orient="records")


def _save_predictions(path: Path, rows: list[dict]) -> None:
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ClassificationError("duplicate ID in predictions rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=PRED_COLUMNS)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _estimate_usd(prompt_chars: int, cfg: dict, output_tokens: int = 12) -> float:
    prices = cfg["budget"].get("price_per_million")
    if not prices:
        return 0.0
    input_tokens = prompt_chars / 4.0
    return (
        input_tokens * float(prices["input"]) / 1_000_000
        + output_tokens * float(prices["output"]) / 1_000_000
    )


def check_approval(out_dir: Path, cfg: dict) -> tuple[bool, str]:
    path = out_dir / "pilot" / "approval.json"
    if not path.exists():
        return False, (
            f"{path} is missing: run the pilot (stage 03) and approve it with "
            "`--approve` before the full run"
        )
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("model") != cfg["model"]["name"]:
        return False, (
            f"approval was granted for model {approval.get('model')!r}, config now "
            f"says {cfg['model']['name']!r}"
        )
    current = prompt_template_sha(cfg)
    if approval.get("prompt_template_sha") != current:
        return False, (
            "prompt template changed since the pilot was approved; re-run the pilot"
        )
    pilot_path = out_dir / "pilot" / "pilot_predictions.csv"
    if pilot_path.exists():
        if approval.get("pilot_predictions_sha256") != file_sha256(pilot_path):
            return False, (
                "pilot predictions changed since approval; review and re-approve"
            )
    return True, f"approved at {approval.get('approved_at')}"


def approve_pilot(config_path: str | Path, note: str = "") -> int:
    cfg = load_config(config_path)
    out_dir = Path(cfg["_meta"]["output_dir"])
    paths = _paths(out_dir, "pilot")
    if not paths["predictions"].exists():
        print(f"ABORT: no pilot predictions at {paths['predictions']}; run the pilot first")
        return 1
    rows = _load_predictions(paths["predictions"])
    ok = sum(1 for row in rows if row["status"] == "ok")
    approval = {
        "approved_at": now_iso(),
        "note": note,
        "model": cfg["model"]["name"],
        "provider": cfg["model"]["provider"],
        "prompt_template_sha": prompt_template_sha(cfg),
        "n_pilot": len(rows),
        "n_ok": ok,
        "pilot_predictions_sha256": file_sha256(paths["predictions"]),
        "temperature_requested": cfg["model"].get("temperature"),
        "temperature_status": "not_requested" if cfg["model"].get("temperature") is None
        else "requested_unverified",
    }
    write_json_atomic(paths["approval"], approval)
    print(f"pilot approved: {paths['approval']}")
    print(f"  model: {approval['model']}")
    print(f"  pilot rows: {len(rows)} ({ok} ok)")
    print(f"  prompt template sha256: {approval['prompt_template_sha']}")
    print(f"  temperature: {approval['temperature_status']}")
    if ok != len(rows):
        print("WARNING: approved with non-ok pilot rows; review them before the full run")
    return 0


def _pilot_comparison(cfg: dict, rows: list[dict]) -> dict | None:
    """Agreement between pilot predictions and a reused prediction file.

    Comparing against earlier *predictions* is allowed; the benchmark stays
    sealed. This is the only sanity check available before a full run.
    """
    predictions_cfg = cfg["dataset"].get("predictions")
    if not predictions_cfg:
        return None
    try:
        reused = read_table(predictions_cfg["path"], predictions_cfg["format"],
                            predictions_cfg["delimiter"], predictions_cfg["encoding"])
    except (FileNotFoundError, ValueError):
        return None
    label_column = predictions_cfg["label_column"]
    id_column = predictions_cfg.get("id_column") or cfg["dataset"].get("id_column")
    if label_column not in reused.columns or id_column not in reused.columns:
        return None
    reused_map = {
        str(row_id).strip(): str(label).strip()
        for row_id, label in zip(reused[id_column], reused[label_column])
    }
    compared = agreed = 0
    disagreements = []
    for row in rows:
        if row["status"] != "ok":
            continue
        previous = reused_map.get(str(row["id"]).strip())
        if previous is None:
            continue
        compared += 1
        if previous == row["label"]:
            agreed += 1
        else:
            disagreements.append({"id": row["id"], "pilot": row["label"],
                                  "reused": previous})
    return {
        "source": str(predictions_cfg["path"]),
        "n_compared": compared,
        "n_agree": agreed,
        "agreement": (agreed / compared) if compared else None,
        "disagreements": disagreements,
        "note": "pilot vs an earlier prediction file, not against the benchmark",
    }


def run(config_path: str | Path, mode: str = "full", limit: int | None = None,
        resume: bool = False, provider=None) -> int:
    from .providers import build_provider

    if mode not in ("pilot", "full"):
        raise ClassificationError(f"mode must be 'pilot' or 'full', got {mode!r}")

    cfg = load_config(config_path)
    out_dir = Path(cfg["_meta"]["output_dir"])
    classes = class_names(cfg)
    blind_path = out_dir / "blind" / "blind.csv"
    blind = _read_blind(blind_path)

    if mode == "pilot":
        n_pilot = int(cfg["pilot"]["n"])
        blind = blind.head(n_pilot).copy()
        print(f"pilot mode: first {len(blind)} observations of the blind order")
    else:
        approved, detail = check_approval(out_dir, cfg)
        if not approved:
            print(f"ABORT: {detail}")
            return 2
        print(f"approval: {detail}")

    paths = _paths(out_dir, mode)
    ids = blind["id"].astype(str).tolist()
    if paths["predictions"].exists() and not resume:
        existing = _load_predictions(paths["predictions"])
        has_results = any(row["status"] != "pending" for row in existing)
        if mode == "full" and has_results:
            raise ClassificationError(
                f"{paths['predictions']} already holds results; pass --resume to "
                "continue it, or remove the file to start over"
            )
        if mode == "pilot" and has_results:
            print("pilot: existing results will be replaced (no --resume given)")
    if paths["predictions"].exists() and resume:
        rows = _load_predictions(paths["predictions"])
        if [row["id"] for row in rows] != ids:
            raise ClassificationError(
                "predictions file IDs do not match the blind input; refusing to resume"
            )
        print(f"resume: {sum(1 for r in rows if r['status'] == 'ok')} row(s) already ok")
    else:
        rows = _init_predictions(paths["predictions"], ids, cfg["model"]["name"])

    provider = provider or build_provider(cfg, workdir=str(blind_path.parent))
    print(f"started: {now_iso()}")
    print(f"provider: {provider.name}  model: {provider.model}")
    print(f"temperature: {provider.temperature_status} "
          f"(requested={provider.temperature!r})")
    print(f"prompt template sha256: {prompt_template_sha(cfg)}")
    print(f"attempt log: {paths['log']}")

    max_calls = int(cfg["budget"]["max_calls"])
    max_usd = cfg["budget"].get("max_usd")
    pilot_calls, full_calls = _total_calls(out_dir)
    calls_used = pilot_calls + full_calls
    estimated_usd = 0.0
    actual_usd = 0.0
    token_totals = {"input": 0, "output": 0, "reasoning": 0,
                    "cache_read": 0, "cache_write": 0}
    timeout_s = int(cfg["model"]["timeout_s"])
    max_attempts = int(cfg["model"]["max_attempts"])
    print(f"budget: {calls_used}/{max_calls} calls already used for this run "
          f"(pilot {pilot_calls} + full {full_calls}); max_usd={max_usd}")

    pending = [row for row in rows if row["status"] != "ok"]
    if limit:
        pending = pending[:limit]
    print(f"rows: {len(rows)} total -> {len(pending)} pending in this run "
          f"(reason: status != 'ok'"
          + (f", limited to {limit}" if limit else "") + ")")

    text_by_id = dict(zip(blind["id"].astype(str), blind["text"].astype(str)))
    locked = lock_sealed(out_dir)
    print(f"sealed directory locked: {locked}")

    stopped = None
    try:
        for index, row in enumerate(pending, 1):
            if stopped:
                break
            text = text_by_id[row["id"]]
            prompt = build_prompt(cfg, text)
            validate_rendering(prompt, cfg, text)
            template_sha = prompt_template_sha(cfg)
            outcome = None
            attempts_used = 0
            for attempt in range(1, max_attempts + 1):
                if calls_used >= max_calls:
                    stopped = "budget_calls_exhausted"
                    break
                projected = max(estimated_usd, actual_usd) + _estimate_usd(len(prompt), cfg)
                if max_usd is not None and projected > float(max_usd):
                    stopped = "budget_usd_exhausted"
                    break

                started = now_iso()
                result = provider.call(prompt, timeout_s)
                calls_used += 1
                attempts_used = attempt
                estimated_usd = projected
                raw = result.raw_stdout or ""
                truncated = len(raw) > RAW_LIMIT
                status, label, detail = "ok", None, ""
                if result.status == "ok" and result.text:
                    status, label, detail = extract_label(result.text, classes)
                else:
                    status = result.status
                    detail = result.error or ""

                record = {
                    "ts_start": started,
                    "ts_end": now_iso(),
                    "run_id": cfg["run"]["run_id"],
                    "mode": mode,
                    "id": row["id"],
                    "attempt": attempt,
                    "provider": provider.name,
                    "model": provider.model,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_template_sha256": template_sha,
                    "temperature_requested": provider.temperature,
                    "temperature_status": provider.temperature_status,
                    "prompt": prompt,
                    "raw_stdout": raw[:RAW_LIMIT],
                    "raw_stdout_truncated": truncated,
                    "raw_stdout_sha256": sha256_text(raw),
                    "raw_stderr": (result.raw_stderr or "")[:4000],
                    "returncode": result.returncode,
                    "elapsed_s": result.elapsed_s,
                    "reply_sha256": sha256_text(result.text) if result.text else None,
                    "status": status,
                    "detail": detail,
                    "parse_method": result.extra.get("parse_method"),
                    "request_body": result.extra.get("request_body"),
                    "usage": result.extra.get("usage"),
                    "cost_usd": result.extra.get("cost_usd"),
                    "estimated_cost_usd": round(_estimate_usd(len(prompt), cfg), 10),
                }
                append_jsonl(paths["log"], record)

                call_cost = result.extra.get("cost_usd")
                if isinstance(call_cost, (int, float)):
                    actual_usd += float(call_cost)
                for token_key, token_value in (result.extra.get("usage") or {}).items():
                    if token_key in token_totals and isinstance(token_value, (int, float)):
                        token_totals[token_key] += int(token_value)

                if status == "ok":
                    row["label"] = label
                    row["status"] = "ok"
                    row["attempts"] = attempt
                    outcome = "ok"
                    break
                outcome = status
                if status.startswith("blocked_"):
                    stopped = status
                    break
                # transient problem: retry up to max_attempts

            if outcome == "ok":
                pass
            elif stopped in TERMINAL_STOP:
                pass  # row stays pending; budget can be raised and the run resumed
            else:
                row["status"] = outcome or "failed_all_attempts"
                row["attempts"] = attempts_used

            _save_predictions(paths["predictions"], rows)
            print(f"[{index}/{len(pending)}] {row['id']}: status={row['status']} "
                  f"label={row['label'] or '-'} attempts={row['attempts']} "
                  f"calls={calls_used}/{max_calls}")
    finally:
        restored = unlock_sealed(out_dir)
        print(f"sealed directory unlocked: {restored}")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"status counts: {counts}")

    # Recompute the split from the logs: the values captured at start describe
    # the budget position before this invocation, not the final spend.
    final_pilot_calls, final_full_calls = _total_calls(out_dir)
    final_calls = final_pilot_calls + final_full_calls
    final_actual, final_estimated, final_tokens = _totals_from_logs(out_dir, cfg)
    if final_calls != calls_used:
        print(f"NOTE: final call count {final_calls} differs from the in-run counter {calls_used}")
    print(f"calls: {final_calls} used in total (pilot {final_pilot_calls} + "
          f"full {final_full_calls}); actual cost ${final_actual:.6f}, "
          f"estimated ${final_estimated:.6f}")

    if mode == "pilot":
        report = {
            "created_at": now_iso(),
            "n_rows": len(rows),
            "n_ok": counts.get("ok", 0),
            "statuses": counts,
            "model": provider.model,
            "prompt_template_sha": prompt_template_sha(cfg),
            "calls_used": final_calls,
            "actual_usd": round(final_actual, 8),
            "estimated_usd": round(final_estimated, 6),
            "token_totals": final_tokens,
            "comparison_to_reused_predictions": _pilot_comparison(cfg, rows),
            "next_step": "review pilot_predictions.csv, then run stage 03 with --approve",
        }
        write_json_atomic(paths["report"], report)
        print(f"pilot report: {paths['report']}")
        comparison = report["comparison_to_reused_predictions"]
        if comparison:
            print(f"pilot vs reused predictions: {comparison['n_agree']}/"
                  f"{comparison['n_compared']} agree")
        print("pilot complete: review the predictions, then approve with "
              "`python scripts/03_pilot.py --config ... --approve`")

    update_manifest(out_dir, "03_pilot" if mode == "pilot" else "04_classify", {
        "finished_at": now_iso(),
        "mode": mode,
        "provider": provider.describe(),
        "prompt_template_sha": prompt_template_sha(cfg),
        "blind_sha256": file_sha256(blind_path),
        "predictions_path": str(paths["predictions"]),
        "predictions_sha256": file_sha256(paths["predictions"]),
        "attempt_log": str(paths["log"]),
        "n_rows": len(rows),
        "status_counts": counts,
        "calls_used_total": final_calls,
        "calls_pilot": final_pilot_calls,
        "calls_full": final_full_calls,
        "max_calls": max_calls,
        "actual_usd": round(final_actual, 8),
        "actual_usd_source": "sum of provider-reported per-call cost from the attempt logs",
        "estimated_usd": round(final_estimated, 6),
        "token_totals": final_tokens,
        "estimated_usd_note": "rough estimate from characters/4 and 12 output tokens; not billing data",
        "sealed_locked_during_run": locked,
        "stopped_reason": stopped,
    })

    if stopped:
        print(f"STOPPED EARLY: {stopped}; remaining rows are pending and resumable")
        return 3
    if counts.get("ok", 0) != len(rows):
        print("RESULT: incomplete; some rows are not ok (see statuses above)")
        return 1
    print("RESULT: all rows classified")
    return 0
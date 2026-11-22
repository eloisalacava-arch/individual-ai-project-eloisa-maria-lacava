"""Token and cost extraction from provider output.

The OpenCode CLI reports authoritative per-call cost and token usage in its
``step_finish`` parts. Keeping the parser in its own module lets the runner
re-derive totals from preserved raw output, including for records written before
this field existed - without rewriting any log.
"""

from __future__ import annotations

import json


def extract_step_usage(raw: str) -> dict | None:
    """Sum CLI-reported cost and token usage across step_finish parts.

    Returns None when the payload carries no step_finish record, so the caller
    can fall back to an estimate rather than reporting a fake zero.
    """
    cost = 0.0
    input_tokens = output_tokens = reasoning_tokens = 0
    cache_read = cache_write = 0
    found = False
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or obj.get("type") != "step_finish":
            continue
        part = obj.get("part")
        if not isinstance(part, dict):
            continue
        found = True
        if isinstance(part.get("cost"), (int, float)):
            cost += float(part["cost"])
        tokens = part.get("tokens")
        if isinstance(tokens, dict):
            input_tokens += int(tokens.get("input") or 0)
            output_tokens += int(tokens.get("output") or 0)
            reasoning_tokens += int(tokens.get("reasoning") or 0)
            cache = tokens.get("cache") or {}
            if isinstance(cache, dict):
                cache_read += int(cache.get("read") or 0)
                cache_write += int(cache.get("write") or 0)
    if not found:
        return None
    return {
        "cost_usd": round(cost, 10),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "cache_read": cache_read,
            "cache_write": cache_write,
        },
    }
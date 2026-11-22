"""Deterministic provider used by tests. Never performs a network or CLI call."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.providers.base import Provider, ProviderResult  # noqa: E402

DEFAULT_EXPLANATION = {
    "summary": "A hypothesis: the two readings differ on emphasis.",
    "reasons": ["mixed_or_conflicting_signals"],
    "evidence": "",
    "what_would_resolve": "more context or per-annotator labels",
    "confidence": "medium",
}


class JsonStubProvider(Provider):
    """Deterministic provider that returns a JSON explanation. No network."""

    name = "json_stub"
    supports_temperature = False

    def __init__(self, cfg, payload=None, raw=None, status="ok",
                 cost_usd=None, usage=None):
        super().__init__(cfg)
        self.payload = payload
        self.raw = raw
        self.status = status
        self.cost_usd = cost_usd
        self.usage = usage
        self.prompts = []
        self.calls = []

    def call(self, prompt, timeout_s):
        import json
        self.prompts.append(prompt)
        self.calls.append(prompt)
        if self.status != "ok":
            return ProviderResult(status=self.status, error=f"stub {self.status}")
        text = self.raw if self.raw is not None else json.dumps(
            self.payload if self.payload is not None else DEFAULT_EXPLANATION)
        return ProviderResult(
            status="ok", text=text, raw_stdout=text, elapsed_s=0.0,
            extra={"parse_method": "json_stub", "cost_usd": self.cost_usd,
                   "usage": self.usage},
        )


class StubProvider(Provider):
    name = "stub"
    supports_temperature = False

    def __init__(self, cfg, label="positive", status="ok", on_call=None,
                 cost_usd=None, usage=None):
        super().__init__(cfg)
        self.label = label
        self.status = status
        self.on_call = on_call
        self.cost_usd = cost_usd
        self.usage = usage
        self.calls = []
        self.prompts = []

    def call(self, prompt, timeout_s):
        self.calls.append(prompt)
        self.prompts.append(prompt)
        if self.on_call:
            self.on_call(self, prompt)
        if self.status != "ok":
            return ProviderResult(status=self.status, error=f"stub {self.status}")
        return ProviderResult(
            status="ok",
            text=f"ANSWER: {self.label}",
            raw_stdout="stub raw output",
            elapsed_s=0.0,
            extra={
                "parse_method": "stub",
                "cost_usd": self.cost_usd,
                "usage": self.usage,
            },
        )
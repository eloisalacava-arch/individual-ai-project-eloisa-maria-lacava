"""OpenAI-compatible HTTP provider.

Portable path for a machine without the OpenCode CLI. The key is read from the
environment variable named in the config; it is never written to a file, a log,
or a manifest. The request body is logged with the prompt, but the
Authorization header is not.

Uses only the standard library so the delivered folder stays light.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import FATAL_STATUS_CODES, Provider, ProviderError, ProviderResult, detect_fatal

LOG_BODY_LIMIT = 100_000


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"
    supports_temperature = True

    def __init__(self, cfg: dict, workdir: str | None = None) -> None:
        super().__init__(cfg, workdir)
        base_url = cfg["model"].get("base_url")
        key_env = cfg["model"].get("api_key_env")
        if not base_url or not key_env:
            raise ProviderError(
                "provider openai_compatible requires model.base_url and model.api_key_env"
            )
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key_env = key_env
        if not os.environ.get(self.key_env):
            raise ProviderError(
                f"environment variable {self.key_env} is not set; export it before "
                "classifying. The key is never read from a config file."
            )

    def _body(self, prompt: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        max_tokens = self.cfg["model"].get("max_output_tokens")
        if max_tokens:
            body["max_tokens"] = max_tokens
        return body

    def call(self, prompt: str, timeout_s: int) -> ProviderResult:
        started = time.time()
        body = self._body(prompt)
        logged_body = json.dumps(body, ensure_ascii=False)
        if len(logged_body) > LOG_BODY_LIMIT:
            logged_body = logged_body[:LOG_BODY_LIMIT] + "...[truncated]"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ[self.key_env]}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # Fatal tokens are scanned only over the error body, never over a
            # successful assistant message, which may legitimately contain words
            # such as "insufficient" or "credit".
            fatal = FATAL_STATUS_CODES.get(exc.code) or detect_fatal(detail)
            status = f"blocked_{fatal}" if fatal else "api_error"
            return ProviderResult(
                status=status,
                raw_stdout=detail[:LOG_BODY_LIMIT],
                raw_stderr="",
                returncode=exc.code,
                elapsed_s=round(time.time() - started, 2),
                error=f"HTTP {exc.code}",
                fatal=fatal,
                extra={"request_body": logged_body},
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            return ProviderResult(
                status="network_error",
                elapsed_s=round(time.time() - started, 2),
                error=f"{type(exc).__name__}: {exc}",
                extra={"request_body": logged_body},
            )

        elapsed = round(time.time() - started, 2)
        try:
            payload = json.loads(raw)
            text = payload["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return ProviderResult(
                status="api_error",
                raw_stdout=raw[:LOG_BODY_LIMIT],
                elapsed_s=elapsed,
                error="response is not an OpenAI-compatible chat completion",
                extra={"request_body": logged_body},
            )

        usage = payload.get("usage") or {}
        cost_usd = None
        prices = self.cfg["budget"].get("price_per_million")
        if usage and prices:
            cost_usd = (
                float(usage.get("prompt_tokens") or 0) * float(prices["input"])
                + float(usage.get("completion_tokens") or 0) * float(prices["output"])
            ) / 1_000_000
        return ProviderResult(
            status="ok",
            text=text,
            raw_stdout=raw[:LOG_BODY_LIMIT],
            elapsed_s=elapsed,
            extra={
                "parse_method": "chat_completion",
                "request_body": logged_body,
                "usage": usage or None,
                "cost_usd": cost_usd,
                "response_model": payload.get("model"),
            },
        )

    def describe(self) -> dict:
        info = super().describe()
        # Name only; the value is never read here and never stored.
        info["api_key_env"] = self.key_env
        info["base_url"] = self.url
        return info
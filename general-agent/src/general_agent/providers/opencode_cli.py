"""Provider for the locally installed OpenCode CLI.

Mirrors the access method of the parent project (``workflow_13_run_classifier.py``):
one fresh session per observation, ``--format json``, ``--archived`` so scripted
runs stay out of session lists, and no credentials handled by us.

The subprocess runs with ``cwd`` set to the blind directory, which contains only
``blind.csv``. This is a practical barrier, not a sandbox; the sealed directory
is additionally locked during classification (see ``blind.lock_sealed``).

No temperature flag exists, so a configured temperature cannot be sent through
this provider. It is reported as ``not_supported`` rather than silently dropped.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ..usage import extract_step_usage
from .base import Provider, ProviderError, ProviderResult, detect_fatal

DEFAULT_CLI = "/Applications/OpenCode.app/Contents/Resources/opencode-cli"


def resolve_cli(cfg: dict) -> str:
    explicit = cfg["model"].get("cli_path")
    candidate = explicit or os.environ.get("OPENCODE_CLI") or DEFAULT_CLI
    if not Path(candidate).exists():
        raise ProviderError(
            f"opencode CLI not found at {candidate!r}. Set model.cli_path in the "
            "config or the OPENCODE_CLI environment variable."
        )
    return candidate


def extract_assistant_text(raw: str, prompt: str) -> tuple[str, str]:
    """Collect structured assistant text parts; fall back to stdout."""
    structured: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        part = obj.get("part") if isinstance(obj, dict) else None
        if (isinstance(part, dict) and part.get("type") == "text"
                and isinstance(part.get("text"), str)):
            structured.append(part["text"])
    stripped_prompt = prompt.strip()
    structured = [part for part in structured if stripped_prompt not in part]
    if structured:
        return "\n".join(structured).strip(), "jsonl_part_text"

    without_prompt = (raw or "").replace(prompt, " ")
    escaped = json.dumps(prompt)[1:-1]
    without_prompt = without_prompt.replace(escaped, " ")
    return without_prompt.strip(), "raw_fallback"


class OpenCodeCliProvider(Provider):
    name = "opencode_cli"
    supports_temperature = False

    def __init__(self, cfg: dict, workdir: str | None = None) -> None:
        super().__init__(cfg, workdir)
        self.cli = resolve_cli(cfg)

    def call(self, prompt: str, timeout_s: int) -> ProviderResult:
        started = time.time()
        command = [
            self.cli, "run",
            "--model", self.model,
            "--format", "json",
            "--archived",
            prompt,
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=self.workdir,
            )
        except subprocess.TimeoutExpired:
            return ProviderResult(
                status="timeout",
                elapsed_s=round(time.time() - started, 2),
                error=f"timeout after {timeout_s}s",
            )
        except OSError as exc:
            return ProviderResult(
                status="local_error",
                elapsed_s=round(time.time() - started, 2),
                error=f"{type(exc).__name__}: {exc}",
            )

        raw = process.stdout or ""
        stderr = (process.stderr or "")[:4000]
        elapsed = round(time.time() - started, 2)
        # Scan diagnostics only, never the assistant text. A legitimate reply can
        # contain words such as "insufficient", "credit", or "balance" (common in
        # financial text and in the reason vocabulary), which must not be read as
        # a provider billing failure. stdout is scanned only when the CLI failed.
        fatal = detect_fatal(stderr)
        if fatal is None and process.returncode != 0:
            fatal = detect_fatal(raw)

        if fatal:
            return ProviderResult(
                status=f"blocked_{fatal}", raw_stdout=raw, raw_stderr=stderr,
                returncode=process.returncode, elapsed_s=elapsed,
                error=f"non-retryable {fatal} error", fatal=fatal,
            )
        if process.returncode != 0:
            return ProviderResult(
                status="api_error", raw_stdout=raw, raw_stderr=stderr,
                returncode=process.returncode, elapsed_s=elapsed,
                error=f"returncode={process.returncode}",
            )

        text, method = extract_assistant_text(raw, prompt)
        if not text:
            return ProviderResult(
                status="empty_reply", raw_stdout=raw, raw_stderr=stderr,
                returncode=process.returncode, elapsed_s=elapsed,
                error="no assistant text in stdout",
                extra={"parse_method": method},
            )
        usage_record = extract_step_usage(raw) or {}
        return ProviderResult(
            status="ok", text=text, raw_stdout=raw, raw_stderr=stderr,
            returncode=process.returncode, elapsed_s=elapsed,
            extra={
                "parse_method": method,
                "cost_usd": usage_record.get("cost_usd"),
                "usage": usage_record.get("tokens"),
            },
        )

    def describe(self) -> dict:
        info = super().describe()
        info["cli_path"] = self.cli
        return info
"""Provider interface and shared error classification.

A provider turns one prompt into one reply. It knows nothing about labels,
benchmarks, metrics, or files other than the input it is given.

Fatal conditions (auth, billing, rate limit) stop a run immediately, as agreed
in the parent project's v1.3 protocol: retrying those burns budget without any
chance of success.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """Raised for configuration problems, before any call is attempted."""


@dataclass
class ProviderResult:
    status: str                     # ok | api_error | timeout | blocked_* | ...
    text: str | None = None         # assistant text, when status == ok
    raw_stdout: str = ""
    raw_stderr: str = ""
    returncode: int | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    fatal: str | None = None        # auth | billing | rate_limit
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


AUTH_TOKENS = (
    "unauthorized", "forbidden", "invalid api key", "authentication",
    "permission denied", "no auth",
)
BILLING_TOKENS = (
    "insufficient", "credit", "billing", "balance", "payment",
    "out of funds", "exceeded your current quota",
)
RATE_TOKENS = (
    "rate limit", "rate-limit", "ratelimit", "too many requests",
    "free-limit", "quota exceeded",
)
FATAL_STATUS_CODES = {401: "auth", 403: "auth", 402: "billing", 429: "rate_limit"}


def detect_fatal(raw: str) -> str | None:
    """Conservative token scan over provider diagnostics.

    Call this on stderr and on error bodies, never on assistant text: a reply may
    legitimately contain "insufficient", "credit", "balance", or "payment" (for
    example a financial explanation), which is not a provider billing failure.
    A bare 'limit' is not fatal (it may mean context length).
    """
    lowered = (raw or "").lower()
    if any(token in lowered for token in AUTH_TOKENS):
        return "auth"
    if any(token in lowered for token in BILLING_TOKENS):
        return "billing"
    if any(token in lowered for token in RATE_TOKENS):
        return "rate_limit"
    return None


class Provider:
    """Base class. Subclasses implement ``call``."""

    name = "base"
    supports_temperature = False

    def __init__(self, cfg: dict, workdir: str | None = None) -> None:
        self.cfg = cfg
        self.workdir = workdir

    @property
    def model(self) -> str:
        return self.cfg["model"]["name"]

    @property
    def temperature(self):
        return self.cfg["model"].get("temperature")

    @property
    def temperature_status(self) -> str:
        if self.temperature is None:
            return "not_requested"
        return "requested_unverified" if self.supports_temperature else "not_supported"

    def call(self, prompt: str, timeout_s: int) -> ProviderResult:
        raise NotImplementedError

    def describe(self) -> dict:
        """Manifest-safe description. Never include credentials."""
        return {
            "provider": self.name,
            "model": self.model,
            "supports_temperature": self.supports_temperature,
            "temperature_requested": self.temperature,
            "temperature_status": self.temperature_status,
        }
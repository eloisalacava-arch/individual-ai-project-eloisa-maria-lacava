"""Provider factory."""

from __future__ import annotations

from .base import Provider, ProviderError, ProviderResult
from .openai_compat import OpenAICompatibleProvider
from .opencode_cli import OpenCodeCliProvider

__all__ = [
    "Provider",
    "ProviderError",
    "ProviderResult",
    "OpenCodeCliProvider",
    "OpenAICompatibleProvider",
    "build_provider",
]


def build_provider(cfg: dict, workdir: str | None = None) -> Provider:
    provider = cfg["model"]["provider"]
    if provider == "opencode_cli":
        return OpenCodeCliProvider(cfg, workdir=workdir)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(cfg, workdir=workdir)
    raise ProviderError(f"unknown provider {provider!r}")
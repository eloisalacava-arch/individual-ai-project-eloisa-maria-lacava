"""Curated provider and model allow-list.

The interface never lets a user type an arbitrary endpoint or a key. It offers
only what ``configs/ui_providers.yaml`` declares, and for key-based providers it
reports whether the named environment variable is present without reading its
value.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from app.workspace import REGISTRY_PATH

FALLBACK = {
    "providers": [
        {
            "id": "opencode",
            "label": "OpenCode CLI (local)",
            "provider": "opencode_cli",
            "enabled": True,
            "access_note": "Uses the signed-in OpenCode account; no key is stored here.",
            "models": [
                {
                    "name": "openrouter/deepseek/deepseek-v4.1-flash-1017",
                    "label": "DeepSeek V4.1 Flash",
                    "price_per_million": {"input": 0.036, "output": 0.50},
                    "max_calls": 500,
                }
            ],
        }
    ]
}


class RegistryError(ValueError):
    """Raised when the registry cannot be used."""


def load_registry(path: Path | None = None) -> dict:
    path = path or REGISTRY_PATH
    if not path.exists():
        return FALLBACK
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("providers"), list):
        raise RegistryError(f"{path.name}: expected a top-level 'providers' list")
    return raw


def enabled_providers(registry: dict | None = None) -> list[dict]:
    registry = registry or load_registry()
    return [p for p in registry["providers"] if p.get("enabled", True) and p.get("models")]


def find_provider(provider_id: str, registry: dict | None = None) -> dict | None:
    for provider in enabled_providers(registry):
        if provider.get("id") == provider_id:
            return provider
    return None


def find_model(provider_id: str, model_name: str,
               registry: dict | None = None) -> dict | None:
    provider = find_provider(provider_id, registry)
    if not provider:
        return None
    for model in provider.get("models", []):
        if model.get("name") == model_name:
            return model
    return None


def credential_status(model_entry: dict, provider_entry: dict) -> tuple[str, bool]:
    """Return (environment variable name, present?). Never the value."""
    if provider_entry.get("provider") != "openai_compatible":
        return "", True
    key_env = provider_entry.get("api_key_env") or ""
    return key_env, bool(key_env) and bool(os.environ.get(key_env))


def missing_key_message(key_env: str) -> str:
    """Clear, non-fatal guidance shown when a key-based provider has no key.

    Returned as plain text; the views wrap it in Streamlit markdown. The exact
    export command is included so it can be copied. This is a warning, not an
    error: validation and sealing still work without a key.
    """
    return (
        f"{key_env} is not set. Validation and sealing still work, but any model "
        "call (classification or qualitative explanations) needs it. Export the "
        "key in the shell that starts Streamlit and reload:  "
        f'export {key_env}="your-key"'
    )


def model_block(provider_entry: dict, model_entry: dict) -> dict:
    """The ``model`` section of the generated config."""
    block = {
        "provider": provider_entry["provider"],
        "name": model_entry["name"],
        "temperature": None,
        "temperature_note": (
            "Not requested by the interface. The OpenCode CLI exposes no "
            "temperature flag; see SPEC.md section 2."
        ),
        "max_output_tokens": None,
        "timeout_s": 180,
        "max_attempts": 3,
        "concurrency": 1,
    }
    if provider_entry["provider"] == "openai_compatible":
        block["base_url"] = provider_entry.get("base_url")
        block["api_key_env"] = provider_entry.get("api_key_env")
    if provider_entry.get("cli_path"):
        block["cli_path"] = provider_entry["cli_path"]
    return block


def price_block(model_entry: dict) -> dict:
    return model_entry.get("price_per_million") or {"input": 0.0, "output": 0.0}


def max_calls_ceiling(model_entry: dict) -> int:
    return int(model_entry.get("max_calls") or 500)

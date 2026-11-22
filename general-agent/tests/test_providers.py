"""Provider adapters: transport parsing, fatal classification, request shape."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from general_agent.config import load_config
from general_agent.providers import build_provider
from general_agent.providers.base import ProviderError, detect_fatal
from general_agent.providers.opencode_cli import extract_assistant_text
from helpers import FIXTURES, make_config


def _openai_cfg(tmp_path, monkeypatch, temperature=None):
    monkeypatch.setenv("GA_TEST_KEY", "sk-not-a-real-key")
    patch = {"model": {
        "provider": "openai_compatible",
        "name": "test/model",
        "base_url": "https://example.invalid/v1",
        "api_key_env": "GA_TEST_KEY",
        "temperature": temperature,
    }}
    return load_config(make_config(tmp_path, patch=patch))


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_opencode_reply_is_extracted_from_structured_parts():
    raw = (FIXTURES / "opencode_raw_reply.jsonl").read_text(encoding="utf-8")
    text, method = extract_assistant_text(raw, prompt="some prompt")
    assert text == "ANSWER: positive"
    assert method == "jsonl_part_text"


def test_prompt_echo_is_excluded_from_assistant_text():
    prompt = "You are a classifier.\nANSWER: <class>"
    raw = "\n".join([
        json.dumps({"part": {"type": "text", "text": prompt}}),
        json.dumps({"part": {"type": "text", "text": "ANSWER: negative"}}),
    ])
    text, _ = extract_assistant_text(raw, prompt)
    assert text == "ANSWER: negative"
    assert prompt not in text


def test_fatal_detection_by_token():
    assert detect_fatal("HTTP 401 Unauthorized") == "auth"
    assert detect_fatal("insufficient credit balance") == "billing"
    assert detect_fatal("rate limit exceeded") == "rate_limit"
    assert detect_fatal("error: context length exceeded its limit") is None


def test_build_provider_rejects_a_missing_cli(tmp_path):
    cfg = load_config(make_config(
        tmp_path, patch={"model": {"cli_path": "/nonexistent/opencode-cli"}}
    ))
    with pytest.raises(ProviderError):
        build_provider(cfg)


def test_openai_provider_requires_the_key_at_construction(tmp_path, monkeypatch):
    monkeypatch.delenv("GA_TEST_KEY", raising=False)
    cfg = load_config(make_config(
        tmp_path,
        patch={"model": {
            "provider": "openai_compatible",
            "name": "test/model",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "GA_TEST_KEY",
        }},
    ))
    with pytest.raises(ProviderError) as excinfo:
        build_provider(cfg)
    assert "GA_TEST_KEY" in str(excinfo.value)


def test_openai_call_sends_prompt_and_returns_content(tmp_path, monkeypatch):
    cfg = _openai_cfg(tmp_path, monkeypatch)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(json.dumps({
            "model": "test-model-v1",
            "choices": [{"message": {"content": "ANSWER: positive"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }).encode("utf-8"))

    monkeypatch.setattr(
        "general_agent.providers.openai_compat.urllib.request.urlopen", fake_urlopen
    )
    provider = build_provider(cfg)
    result = provider.call("Classify this observation.", timeout_s=30)

    assert result.ok
    assert result.text == "ANSWER: positive"
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["body"]["model"] == "test/model"
    assert captured["body"]["messages"][0]["content"] == "Classify this observation."
    assert "temperature" not in captured["body"]
    assert "Authorization" in captured["headers"]
    assert "sk-not-a-real-key" not in json.dumps(result.extra["usage"])


def test_openai_temperature_is_sent_when_configured(tmp_path, monkeypatch):
    cfg = _openai_cfg(tmp_path, monkeypatch, temperature=0)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(json.dumps({
            "choices": [{"message": {"content": "ANSWER: negative"}}]
        }).encode("utf-8"))

    monkeypatch.setattr(
        "general_agent.providers.openai_compat.urllib.request.urlopen", fake_urlopen
    )
    provider = build_provider(cfg)
    assert provider.temperature_status == "requested_unverified"
    provider.call("prompt", timeout_s=30)
    assert captured["body"]["temperature"] == 0


def test_openai_rate_limit_is_fatal(tmp_path, monkeypatch):
    cfg = _openai_cfg(tmp_path, monkeypatch)

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", hdrs=None,
            fp=io.BytesIO(b'{"error":"rate limit exceeded"}'),
        )

    monkeypatch.setattr(
        "general_agent.providers.openai_compat.urllib.request.urlopen", fake_urlopen
    )
    result = build_provider(cfg).call("prompt", timeout_s=30)
    assert result.status == "blocked_rate_limit"
    assert result.fatal == "rate_limit"


def test_openai_non_json_response_is_an_api_error(tmp_path, monkeypatch):
    cfg = _openai_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "general_agent.providers.openai_compat.urllib.request.urlopen",
        lambda request, timeout: _FakeResponse(b"<html>gateway error</html>"),
    )
    result = build_provider(cfg).call("prompt", timeout_s=30)
    assert result.status == "api_error"
    assert result.text is None


def test_opencode_declares_no_temperature_support():
    from general_agent.providers.opencode_cli import OpenCodeCliProvider

    assert OpenCodeCliProvider.supports_temperature is False


def test_temperature_status_reflects_provider_support(tmp_path):
    cfg = load_config(make_config(tmp_path, patch={"model": {"temperature": 0}}))

    from general_agent.providers.base import Provider

    class Supported(Provider):
        name = "supported"
        supports_temperature = True

    class Unsupported(Provider):
        name = "unsupported"
        supports_temperature = False

    assert Supported(cfg).temperature_status == "requested_unverified"
    assert Unsupported(cfg).temperature_status == "not_supported"

def _fake_cli(tmp_path):
    path = tmp_path / "fake-opencode"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_opencode_provider_builds_the_expected_command(tmp_path, monkeypatch):
    import subprocess

    from general_agent.config import load_config
    from general_agent.providers import build_provider

    cli = _fake_cli(tmp_path)
    cfg = load_config(make_config(tmp_path, patch={"model": {"cli_path": str(cli)}}))
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, 0,
            stdout='{"part":{"type":"text","text":"ANSWER: positive"}}\n',
            stderr="",
        )

    monkeypatch.setattr(
        "general_agent.providers.opencode_cli.subprocess.run", fake_run
    )
    provider = build_provider(cfg, workdir=str(tmp_path))
    result = provider.call("Classify this observation.", timeout_s=42)

    assert result.ok
    assert result.text == "ANSWER: positive"
    assert recorded["command"] == [
        str(cli), "run", "--model", "test/model", "--format", "json",
        "--archived", "Classify this observation.",
    ]
    assert recorded["kwargs"]["cwd"] == str(tmp_path)
    assert recorded["kwargs"]["timeout"] == 42


def test_opencode_timeout_is_reported_not_raised(tmp_path, monkeypatch):
    import subprocess

    from general_agent.config import load_config
    from general_agent.providers import build_provider

    cli = _fake_cli(tmp_path)
    cfg = load_config(make_config(tmp_path, patch={"model": {"cli_path": str(cli)}}))

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(
        "general_agent.providers.opencode_cli.subprocess.run", fake_run
    )
    result = build_provider(cfg, workdir=str(tmp_path)).call("prompt", timeout_s=5)
    assert result.status == "timeout"
    assert result.text is None


def test_opencode_billing_error_is_fatal(tmp_path, monkeypatch):
    import subprocess

    from general_agent.config import load_config
    from general_agent.providers import build_provider

    cli = _fake_cli(tmp_path)
    cfg = load_config(make_config(tmp_path, patch={"model": {"cli_path": str(cli)}}))

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="402 insufficient credit balance"
        )

    monkeypatch.setattr(
        "general_agent.providers.opencode_cli.subprocess.run", fake_run
    )
    result = build_provider(cfg, workdir=str(tmp_path)).call("prompt", timeout_s=5)
    assert result.status == "blocked_billing"
    assert result.fatal == "billing"


def test_opencode_success_with_billing_words_in_the_reply_is_not_fatal(tmp_path,
                                                                       monkeypatch):
    """A valid reply mentioning 'insufficient' must not be read as a billing block."""
    import subprocess

    from general_agent.config import load_config
    from general_agent.providers import build_provider

    cli = _fake_cli(tmp_path)
    cfg = load_config(make_config(tmp_path, patch={"model": {"cli_path": str(cli)}}))
    explanation = json.dumps({
        "summary": "The historical loss may be baseline data rather than a new loss.",
        "reasons": ["insufficient_context_in_observation",
                    "category_definition_overlap"],
        "evidence": "Net loss in the same period in 2009",
        "what_would_resolve": "surrounding sentences",
        "confidence": "medium",
    })
    raw = "\n".join([
        json.dumps({"type": "text", "part": {"type": "text", "text": explanation}}),
        json.dumps({"type": "step_finish", "part": {
            "type": "step-finish", "cost": 0.0004,
            "tokens": {"input": 100, "output": 20, "cache": {"read": 0, "write": 0}}}}),
    ])

    monkeypatch.setattr(
        "general_agent.providers.opencode_cli.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, raw, ""),
    )
    result = build_provider(cfg).call("prompt", timeout_s=5)
    assert result.status == "ok"
    assert "insufficient_context_in_observation" in result.text
    assert result.extra["cost_usd"] == pytest.approx(0.0004)


def test_opencode_stderr_fatal_is_blocked_even_with_zero_returncode(tmp_path,
                                                                    monkeypatch):
    import subprocess

    from general_agent.config import load_config
    from general_agent.providers import build_provider

    cli = _fake_cli(tmp_path)
    cfg = load_config(make_config(tmp_path, patch={"model": {"cli_path": str(cli)}}))
    monkeypatch.setattr(
        "general_agent.providers.opencode_cli.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "", "402 insufficient credit balance"),
    )
    result = build_provider(cfg).call("prompt", timeout_s=5)
    assert result.status == "blocked_billing"
    assert result.fatal == "billing"


def test_opencode_nonzero_returncode_with_fatal_stdout_is_blocked(tmp_path,
                                                                  monkeypatch):
    import subprocess

    from general_agent.config import load_config
    from general_agent.providers import build_provider

    cli = _fake_cli(tmp_path)
    cfg = load_config(make_config(tmp_path, patch={"model": {"cli_path": str(cli)}}))
    monkeypatch.setattr(
        "general_agent.providers.opencode_cli.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "rate limit exceeded", ""),
    )
    result = build_provider(cfg).call("prompt", timeout_s=5)
    assert result.status == "blocked_rate_limit"


def test_openai_success_content_with_billing_words_is_not_fatal(tmp_path, monkeypatch):
    cfg = _openai_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "general_agent.providers.openai_compat.urllib.request.urlopen",
        lambda request, timeout: _FakeResponse(json.dumps({
            "choices": [{"message": {
                "content": "insufficient credit balance in the comparison period"}}],
        }).encode("utf-8")),
    )
    result = build_provider(cfg).call("prompt", timeout_s=30)
    assert result.status == "ok"
    assert "insufficient" in result.text


def test_opencode_reports_cost_and_tokens_from_step_finish(tmp_path, monkeypatch):
    import subprocess

    from general_agent.config import load_config
    from general_agent.providers import build_provider

    cli = _fake_cli(tmp_path)
    cfg = load_config(make_config(tmp_path, patch={"model": {"cli_path": str(cli)}}))
    raw = (FIXTURES / "opencode_raw_reply.jsonl").read_text(encoding="utf-8")

    monkeypatch.setattr(
        "general_agent.providers.opencode_cli.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, raw, ""),
    )
    result = build_provider(cfg).call("prompt", timeout_s=5)
    assert result.extra["cost_usd"] == pytest.approx(0.000080472)
    assert result.extra["usage"]["input"] == 317
    assert result.extra["usage"]["output"] == 4
    assert result.extra["usage"]["cache_read"] == 6656

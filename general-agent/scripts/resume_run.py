#!/usr/bin/env python
"""Resume a UI run through the simple pipeline without new model calls.

For a run whose pilot and classification are already complete, this drives
``app.pipeline.advance`` from the stored authorization with the qualitative
stage disabled, then finishes evaluation, the statistical report, and the study
PDF. A provider that raises on any ``call`` is injected as a tripwire, so if the
pipeline ever tried to reach a model the script would fail loudly instead of
spending budget. No frozen artefact is modified and the engine is untouched.

Usage:
    uv run python scripts/resume_run.py --run-id simple_interface_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import pipeline, workspace  # noqa: E402
from general_agent.config import load_config  # noqa: E402
from general_agent.providers.base import Provider, ProviderResult  # noqa: E402

TERMINAL = {"done", "stopped", "error", "blocked", "needs_pilot_review"}


class NoCallProvider(Provider):
    """A tripwire: refuses to reach a model so a resume cannot spend budget."""

    name = "no_call_guard"
    supports_temperature = False

    def call(self, prompt: str, timeout_s: int) -> ProviderResult:
        raise RuntimeError(
            "A model call was attempted during a resume that must make none.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=60,
                        help="safety cap on the number of pipeline units")
    args = parser.parse_args()

    run_id = args.run_id
    run_dir = workspace.run_dir(run_id)
    cfg_path = run_dir / "study.yaml"
    if not cfg_path.exists():
        print(f"[resume] no study config at {cfg_path}")
        return 2

    stored = workspace.read_authorization(run_id) or {}
    authorization = dict(stored)
    # Skip qualitative for this resume: no explanations.csv exists and we make
    # no qualitative model calls. The stored authorization is left untouched.
    authorization["qualitative"] = {"enabled": False}
    authorization.setdefault("allow_unclean_pilot", False)

    guard = NoCallProvider(load_config(cfg_path))

    for _ in range(args.limit):
        advance = pipeline.advance(
            run_id, cfg_path, authorization,
            classification_provider=guard,
            qualitative_provider_factory=None)
        print(f"[resume] status={advance.status} stage={advance.stage} "
              f"action={advance.action!r} detail={advance.detail!r}")
        if advance.status in TERMINAL:
            return 0 if advance.status == "done" else 1
    print("[resume] limit reached without a terminal status")
    return 1


if __name__ == "__main__":
    sys.exit(main())

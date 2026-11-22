#!/usr/bin/env python
"""Stage 05 - evaluate frozen predictions against the sealed reference.

Writes every table under runs/<run_id>/results/. Freezes predictions before the
benchmark is opened; the order is recorded in results/evaluation_log.txt.

Usage:
    python scripts/05_evaluate.py --config configs/case2_phrasebank.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))

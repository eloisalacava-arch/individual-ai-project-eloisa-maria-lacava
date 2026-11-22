#!/usr/bin/env python
"""Stage 06 - descriptive disagreement analysis.

Reads the frozen predictions, the sealed reference, and the blind text; writes
t11-t13 under runs/<run_id>/results/. Does not modify the stage-05 tables.

Usage:
    python scripts/06_disagreements.py --config configs/case2_phrasebank.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["disagreements", *sys.argv[1:]]))

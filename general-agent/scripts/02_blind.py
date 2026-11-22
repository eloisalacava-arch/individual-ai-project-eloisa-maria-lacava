#!/usr/bin/env python
"""Stage 02 - build the blind input and seal the benchmark.

No model calls. Writes runs/<run_id>/blind/blind.csv (id, text only) and
runs/<run_id>/sealed/benchmark.csv, applies the seeded unstratified sample, and
records every row count and hash in the manifest.

Usage:
    python scripts/02_blind.py --config configs/case2_phrasebank.yaml
    python scripts/02_blind.py --config configs/case2_phrasebank.yaml --unlock
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["blind", *sys.argv[1:]]))
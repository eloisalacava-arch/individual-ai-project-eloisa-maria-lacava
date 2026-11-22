#!/usr/bin/env python
"""Stage 01 - audit dataset and benchmark, propose or confirm the mapping.

No model calls. Writes runs/<run_id>/{decisions.json,audit.txt,manifest.json}.
Exit codes: 0 PASS, 1 FAIL, 2 CLARIFICATION NEEDED.

Usage:
    python scripts/01_audit.py --config configs/imdb_case_study.yaml
    python scripts/01_audit.py --config configs/case2_phrasebank.yaml --strict
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["audit", *sys.argv[1:]]))
#!/usr/bin/env python
"""Stage 07 - cross-dataset comparison, figures, and generated report.

Reads only per-run result tables. The same code handles any number of configs.

Usage:
    python scripts/07_compare.py --configs configs/imdb_case_study.yaml \
        configs/case2_phrasebank.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["compare", *sys.argv[1:]]))

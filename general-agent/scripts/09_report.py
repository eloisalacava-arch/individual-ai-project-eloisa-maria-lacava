#!/usr/bin/env python
"""Stage 09 - compose per-study and comparison reports, and export PDFs.

Reads the existing result and qualitative tables only. No model call, no
statistic recomputed, no frozen artefact modified.

Usage:
    python scripts/09_report.py --config configs/my_study.yaml
    python scripts/09_report.py --run-id imdb_case_study
    python scripts/09_report.py --all
    python scripts/09_report.py --comparison
    python scripts/09_report.py --config configs/my_study.yaml --no-pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["report", *sys.argv[1:]]))

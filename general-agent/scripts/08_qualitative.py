#!/usr/bin/env python
"""Stage 08 - qualitative disagreement analysis.

Reads the frozen predictions and the sealed reference (never before stage 05 has
frozen), explains a seeded sample of disagreements and a control sample of
agreements, aggregates recurring hypotheses, and writes a narrative report.

Explanations are hypotheses only. A pilot must be approved before any full run.

Usage:
    python scripts/08_qualitative.py --config configs/study.yaml --settings runs/<id>/qualitative/settings.yaml --pilot
    python scripts/08_qualitative.py --config configs/study.yaml --settings ... --approve
    python scripts/08_qualitative.py --config configs/study.yaml --settings ... --run
    python scripts/08_qualitative.py --config configs/study.yaml --settings ... --report
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["qualitative", *sys.argv[1:]]))

#!/usr/bin/env python
"""Stage 04 - full classification run.

Requires an approved pilot (stage 03). Reads only blind/blind.csv, saves
atomically after each row, honours budget caps, and can be resumed:

    python scripts/04_classify.py --config ... --resume
    python scripts/04_classify.py --config ... --limit 50    # partial batch

Exit codes: 0 all rows ok, 1 incomplete, 2 approval/config missing,
3 stopped early (budget or fatal provider error). Remaining rows stay pending.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["classify", *sys.argv[1:]]))
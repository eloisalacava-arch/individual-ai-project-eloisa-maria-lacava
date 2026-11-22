#!/usr/bin/env python
"""Stage 03 - pilot classification and the approval checkpoint.

Two invocations:

    python scripts/03_pilot.py --config ...            # classify pilot.n rows
    python scripts/03_pilot.py --config ... --approve  # record approval

The full run (stage 04) refuses to start without an approval file whose model
and prompt-template hash still match the current config.

Usage notes:
  * the pilot writes pilot/pilot_predictions.csv, never the main predictions file;
  * if the config reuses an earlier prediction file, the pilot report also shows
    agreement against it (predictions only - the benchmark stays sealed);
  * exit codes: 0 ok, 1 failure, 2 needs approval/config attention,
    3 stopped early (budget or fatal provider error).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["pilot", *sys.argv[1:]]))
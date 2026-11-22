#!/usr/bin/env python
"""Stage 00 - validate a study config, resolve paths, record defaults.

Thin wrapper over ``general_agent.cli configure`` so the numbered-script
convention matches the parent project (00_..., 01_...).

Usage:
    python scripts/00_configure.py --config configs/imdb_case_study.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from general_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["configure", *sys.argv[1:]]))
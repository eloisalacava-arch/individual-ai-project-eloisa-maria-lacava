"""Make the UI package importable from tests.

pytest already puts ``src`` on the path (see pyproject). The interface lives in
``general-agent/app``, so the project root must be importable too.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

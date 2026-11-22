"""Composed reporting: per-study reports, comparison PDF, and PDF export.

Presentation only. Every number is read from ``results/`` or
``qualitative/tables/``; no statistic is recomputed and no frozen artefact is
modified.
"""

from __future__ import annotations

from . import build, compose, figures, paths, pdf  # noqa: F401

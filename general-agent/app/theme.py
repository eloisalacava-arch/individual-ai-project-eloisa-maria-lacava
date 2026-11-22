"""Visual theme and small presentation helpers.

One accent colour, generous spacing, and no emoji. The interface is meant to
look like a quiet research tool rather than a dashboard demo.
"""

from __future__ import annotations

ACCENT = "#2F5D8A"
MUTED = "#6B7280"
BORDER = "#E5E7EB"

CUSTOM_CSS = f"""
<style>
  .block-container {{ padding-top: 2.0rem; padding-bottom: 3.5rem; max-width: 1180px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .ga-subtitle {{ color: {MUTED}; font-size: 0.98rem; margin-top: -0.6rem; }}
  .ga-muted {{ color: {MUTED}; font-size: 0.85rem; }}
  .ga-card {{
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 0.9rem 1.1rem;
      background: #FFFFFF;
      margin-bottom: 0.8rem;
  }}
  .ga-chip {{
      display: inline-block;
      border-radius: 999px;
      padding: 0.12rem 0.6rem;
      font-size: 0.78rem;
      border: 1px solid {BORDER};
      color: {MUTED};
      margin-right: 0.35rem;
  }}
  .ga-chip-done {{ border-color: #B7D3B0; color: #3F6B3A; background: #F2F8F1; }}
  .ga-chip-current {{ border-color: {ACCENT}; color: {ACCENT}; background: #F1F6FB; }}
  .ga-metric-source {{ color: {MUTED}; font-size: 0.78rem; }}
</style>
"""


def chip(label: str, state: str = "") -> str:
    """A tiny inline status label: done, current, or plain."""
    klass = "ga-chip"
    if state == "done":
        klass += " ga-chip-done"
    elif state == "current":
        klass += " ga-chip-current"
    return f'<span class="{klass}">{label}</span>'


def status_word(code: int) -> tuple[str, str]:
    """Map an engine exit code to a Streamlit status kind and a short word."""
    if code == 0:
        return "success", "completed"
    if code == 2:
        return "warning", "blocked by a precondition"
    if code == 3:
        return "warning", "stopped early"
    return "error", "failed"

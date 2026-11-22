"""The Markdown-subset parser and the PDF renderer."""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

from general_agent.report import pdf


def test_parse_blocks_handles_the_supported_subset():
    markdown = (
        "# Title\n\n"
        "Intro with & and < angle > brackets.\n\n"
        "## 1. Section\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "- one\n- two\n\n"
        "> a callout\n\n"
        "```\nuv run example\n```\n\n"
        "<!-- figure: key | A caption -->\n"
    )
    blocks = pdf.parse_blocks(markdown)
    kinds = [kind for kind, _ in blocks]
    for kind in ("heading", "paragraph", "table", "list", "quote", "code", "figure"):
        assert kind in kinds, kind
    table = next(payload for kind, payload in blocks if kind == "table")
    assert table == [["a", "b"], ["1", "2"]]  # the separator row is dropped
    figure = next(payload for kind, payload in blocks if kind == "figure")
    assert figure == ("key", "A caption")


def test_markdown_to_pdf_writes_a_pdf(tmp_path):
    markdown = (
        "# Report\n\n"
        "## 1. First\n\nText with unicode ≥ − “quotes” & <tags> and a list.\n\n"
        "| Metric | Value |\n|---|---|\n| Accuracy | 0.900 |\n\n"
        "- one\n- two\n\n> A callout.\n"
    )
    out = tmp_path / "report.pdf"
    pdf.markdown_to_pdf(markdown, out, title="Report", hypotheses=True)
    data = out.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 800


def test_an_unclosed_table_does_not_crash(tmp_path):
    markdown = "# R\n\n## 1. S\n\n| a | b\n| 1 | 2\n\n- item\n"
    out = tmp_path / "r.pdf"
    pdf.markdown_to_pdf(markdown, out)
    assert out.read_bytes().startswith(b"%PDF-")


def test_safe_maps_glyphs_outside_the_pdf_encoding():
    assert pdf._safe("a ≥ b − c × d → e") == "a >= b - c x d -> e"
    assert pdf._safe("dash – and — quotes “ ” ‘ ’") == 'dash - and - quotes " " \' \''

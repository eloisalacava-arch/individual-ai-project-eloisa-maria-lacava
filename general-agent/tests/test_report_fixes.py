"""Regression tests for the six academic-report corrections.

1. A pre-specified subgroup is not called significant once Benjamini-Hochberg
   correction puts q at or above 0.05.
2. The prediction-bias sentence states the direction ("less often").
3. The disagreement-structure shares name their denominator (all analysed rows).
4. The qualitative narrative never claims the statistical analysis is missing.
5. Wide subgroup tables fit the frame with usable columns, and qualitative table
   headers are readable (no long snake_case tokens).
6. The comparison reproduction command has no trailing backslash.

No model call and no statistic recomputed: report text is composed from
hand-written tables that mirror the frozen ones.
"""

from __future__ import annotations

import pandas as pd
import pytest

from general_agent.evaluate.subgroups import verdict_reading
from general_agent.report import compose, pdf
from test_compare import _study


# --------------------------------------------------------------------------- #
# 1. Multiplicity in the subgroup reading
# --------------------------------------------------------------------------- #

def test_verdict_reading_respects_multiplicity():
    supported = verdict_reading("lower_accuracy_significant", 0.019,
                                "pre_specified")
    assert "significantly lower accuracy" in supported

    downgraded = verdict_reading("lower_accuracy_significant", 0.055,
                                 "pre_specified")
    assert "not significant after Benjamini-Hochberg correction" in downgraded
    assert "significantly lower" not in downgraded
    assert "q = 0.055" in downgraded

    # A composite carries no q, so the base reading is kept.
    assert verdict_reading("lower_accuracy_significant", float("nan"),
                           "composite") == supported


def test_subgroup_report_table_downgrades_a_row_whose_q_does_not_clear_half(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame([{
        "reference": "imdb_label", "subgroup": "human_mixed",
        "family": "pre_specified", "n_subgroup": 50, "errors_subgroup": 6,
        "n_reference": 106, "errors_reference": 2, "rate_subgroup": 0.12,
        "rate_reference": 0.019, "risk_difference": 0.101, "rd_ci_low": 0.022,
        "rd_ci_high": 0.220, "fisher_p": 0.014, "bh_q": 0.055,
        "verdict": "lower_accuracy_significant",
    }]).to_csv(results / "t14_subgroup_tests.csv", index=False)

    cfg = {
        "benchmark": {"label_column": "human_sentiment",
                      "reference_labels": {
                          "imdb_label": "star-derived IMDb proxy label"}},
        "report": {},
    }
    lines, verdict = compose._subgroup_section(cfg, results)
    text = "\n".join(lines)
    assert verdict is None
    assert "not significant after Benjamini-Hochberg correction" in text
    assert "significantly lower accuracy for the subgroup" not in text


# --------------------------------------------------------------------------- #
# 2 and 3. Interpretation wording and share denominator
# --------------------------------------------------------------------------- #

def _classes_three() -> pd.DataFrame:
    return pd.DataFrame({
        "class": ["negative", "neutral", "positive"],
        "n_reference": [38, 182, 80], "n_predicted": [44, 183, 73],
        "precision": [0.841, 0.934, 0.904],
        "recall": [0.974, 0.940, 0.825],
        "recall_ci_low": [0.865, 0.895, 0.727],
        "recall_ci_high": [0.995, 0.966, 0.893],
        "f1": [0.902, 0.937, 0.863],
        "f1_ci_low": [0.825, 0.910, 0.803],
        "f1_ci_high": [0.962, 0.961, 0.916],
    })


def _row_three() -> dict:
    return {
        "accuracy": 0.913, "balanced_accuracy": 0.913, "kappa_unweighted": 0.841,
        "n": 300, "n_classes": 3, "n_disagreements": 26,
        "majority_class": "neutral", "majority_share": 0.607,
    }


def test_interpretation_states_the_prediction_bias_direction():
    distribution = pd.DataFrame({
        "class": ["negative", "neutral", "positive"],
        "difference_ai_minus_human": [0.020, 0.003, -0.023],
    })
    severity = pd.DataFrame({"bucket": ["agree", "adjacent", "opposite"],
                             "n": [274, 23, 3], "share": [0.913, 0.077, 0.010]})
    text = "\n".join(compose._interpretation(
        _row_three(), _classes_three(), severity, distribution, True, 3, False,
        None))
    bias_line = next(line for line in text.splitlines() if "AI predicted" in line)
    assert "2.3 percentage points less often" in bias_line
    assert "more often" not in bias_line


def test_interpretation_names_the_share_denominator_for_ordinal_buckets():
    distribution = pd.DataFrame({
        "class": ["negative", "neutral", "positive"],
        "difference_ai_minus_human": [0.020, 0.003, -0.023],
    })
    severity = pd.DataFrame({"bucket": ["agree", "adjacent", "opposite"],
                             "n": [274, 23, 3], "share": [0.913, 0.077, 0.010]})
    text = "\n".join(compose._interpretation(
        _row_three(), _classes_three(), severity, distribution, True, 3, False,
        None))
    assert "Of all 300 analysed rows" in text
    assert "7.7% (23) were adjacent disagreements" in text
    assert "1.0% (3) were opposite" in text


def test_disagreement_structure_states_the_share_denominator():
    severity = pd.DataFrame({"bucket": ["agree", "adjacent", "opposite"],
                             "n": [274, 23, 3], "share": [0.913, 0.077, 0.010]})
    pairs = pd.DataFrame({"reference": ["positive"], "predicted": ["neutral"],
                          "n": [11], "share_of_disagreements": [0.423]})
    text = "\n".join(compose._disagreement_structure(severity, pairs, 26, True, 3))
    assert "share of all 300 analysed rows" in text
    assert "not of the 26 disagreements" in text


# --------------------------------------------------------------------------- #
# 4. The qualitative narrative never claims missing statistics
# --------------------------------------------------------------------------- #

def test_narrative_points_to_the_statistical_report_without_claiming_it_missing(tmp_path):
    from general_agent.qualitative import narrative

    out = tmp_path / "runs" / "r1" / "qualitative"
    out.mkdir(parents=True)
    text = "\n".join(narrative.statistical_lines(out))
    assert "was not generated" not in text
    assert "analysis is missing" in text

    # A legacy per-run statistical report is still embedded when it exists.
    legacy = out.parent / "report"
    legacy.mkdir()
    (legacy / "report.md").write_text("# Study report: x\n\nbody", encoding="utf-8")
    assert "Study report" in "\n".join(narrative.statistical_lines(out))


# --------------------------------------------------------------------------- #
# 5. Readability: wide tables and qualitative headers
# --------------------------------------------------------------------------- #

def test_wide_table_widths_fit_the_frame_and_keep_columns_usable():
    rows = [["Reference", "Subgroup", "Family", "Subgroup n/err",
             "Reference n/err", "Rate (subgroup)", "Rate (reference)",
             "Risk diff (pp)", "95% CI (pp)", "Fisher p", "BH q", "Reading"]]
    rows.append([
        "star-derived IMDb proxy label", "human_mixed", "pre_specified", "50/6",
        "106/2", "12.0%", "1.9%", "+10.1", "+2.2 to +22.0", "0.014", "0.055",
        "lower accuracy for the subgroup with an interval excluding zero, but "
        "not significant after Benjamini-Hochberg correction (q = 0.055)"])
    available = 595.27 - 2 * 51
    widths = pdf.table_column_widths(rows, available)
    assert len(widths) == 12
    assert sum(widths) == pytest.approx(available)
    assert min(widths) >= 0.3 * pdf.POINTS_PER_INCH


def test_pdf_renders_a_wide_table(tmp_path):
    pytest.importorskip("reportlab")
    header = ("| Reference | Subgroup | Family | Subgroup n/err | Reference n/err "
              "| Rate (subgroup) | Rate (reference) | Risk diff (pp) | 95% CI (pp) "
              "| Fisher p | BH q | Reading |")
    separator = "|" + "---|" * 12
    body = ("| star-derived proxy | human_mixed | pre_specified | 50/6 | 106/2 "
            "| 12.0% | 1.9% | +10.1 | +2.2 to +22.0 | 0.014 | 0.055 | not "
            "significant after correction |")
    markdown = f"# R\n\n## 1. S\n\n{header}\n{separator}\n{body}\n"
    out = tmp_path / "r.pdf"
    pdf.markdown_to_pdf(markdown, out)
    assert out.read_bytes().startswith(b"%PDF-")


def _study_with_qualitative(tmp_path, run_id="ui_study"):
    from general_agent.qualitative.run import report as qualitative_report
    from qualitative_helpers import evaluated_run, run_qualitative
    study = evaluated_run(tmp_path, classes=3, n=15, run_id=run_id)
    sidecar, _settings = run_qualitative(
        study, tmp_path, sample={"n": 4, "control_n": 2},
        pilot={"disagreements": 2, "agreements": 1}, budget={"max_calls": 15})
    assert qualitative_report(study["config_path"], sidecar) == 0
    return study


def test_qualitative_table_headers_are_readable(tmp_path):
    study = _study_with_qualitative(tmp_path)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "share_of_explained" not in markdown
    assert "share_of_selected" not in markdown
    assert "difference_disagreements_minus_control" not in markdown
    assert "Share of explained" in markdown
    assert "Difference (disagreements minus control)" in markdown


# --------------------------------------------------------------------------- #
# 6. Reproduction command
# --------------------------------------------------------------------------- #

def test_comparison_reproducibility_command_has_no_trailing_backslash(tmp_path):
    from general_agent.evaluate.analysis import run_evaluation
    from general_agent.evaluate.compare import run_comparison
    from general_agent.evaluate.disagreements import run_disagreements

    first = _study(tmp_path / "a", "study_a", classes=3, error_offset=4)
    second = _study(tmp_path / "b", "study_b", classes=2, error_offset=3)
    for config_path in (first, second):
        assert run_evaluation(config_path) == 0
        assert run_disagreements(config_path) == 0

    out = tmp_path / "comparison"
    assert run_comparison([first, second], output_dir=out) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    block = report.split("Reproducibility", 1)[1].split("```", 2)[1]
    lines = [line for line in block.strip().splitlines() if line.strip()]
    assert any(line.rstrip().endswith("\\") for line in lines)
    assert not lines[-1].rstrip().endswith("\\")

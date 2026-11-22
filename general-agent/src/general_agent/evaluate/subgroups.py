"""Stage 06 helper - pre-specified subgroup and three-way reference analysis.

This module turns the binary subgroup columns already carried in the sealed
benchmark (for example a study's difficulty flags) into an explicit hypothesis
test, and decomposes the relationship between the AI prediction and the primary
and secondary references (the primary benchmark and any declared extra
reference).

Everything here is descriptive and observational. A higher error rate for a
subgroup is an association on one sample, never a causal effect, and the
pre-specified flags are the only ones treated as a confirmatory family; any
composite is reported as a single exploratory contrast.

Statistics:
  * Wilson score interval for each proportion;
  * Newcombe hybrid-score interval for the risk difference;
  * two-sided Fisher exact test for the 2x2 table;
  * Benjamini-Hochberg q-values across the pre-specified flags only.
"""

from __future__ import annotations

import math
from math import comb
from pathlib import Path

import pandas as pd

from .intervals import wilson


def newcombe_risk_difference(k1: int, n1: int, k0: int, n0: int,
                             level: float = 0.95) -> tuple[float, float, float]:
    """Risk difference ``p1 - p0`` with a Newcombe hybrid-score interval."""
    if n1 == 0 or n0 == 0:
        return (math.nan, math.nan, math.nan)
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1, level)
    l0, u0 = wilson(k0, n0, level)
    rd = p1 - p0
    low = rd - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    high = rd + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return (rd, max(-1.0, low), min(1.0, high))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the table ``[[a, b], [c, d]]``."""
    n = a + b + c + d
    if n == 0:
        return math.nan
    row1, col1 = a + b, a + c

    def probability(x: int) -> float:
        return (comb(row1, x) * comb(n - row1, col1 - x)) / comb(n, col1)

    observed = probability(a)
    low = max(0, col1 - (n - row1))
    high = min(row1, col1)
    total = 0.0
    for x in range(low, high + 1):
        p = probability(x)
        if p <= observed + 1e-12:
            total += p
    return min(1.0, total)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Step-up q-values in the input order; NaN inputs stay NaN."""
    indexed = [(i, float(p)) for i, p in enumerate(pvalues)
               if p is not None and not math.isnan(float(p))]
    qvalues = [math.nan] * len(pvalues)
    m = len(indexed)
    if m == 0:
        return qvalues
    ordered = sorted(indexed, key=lambda item: item[1])
    previous = 1.0
    for position in range(m, 0, -1):
        index, p = ordered[position - 1]
        value = min(previous, p * m / position)
        qvalues[index] = value
        previous = value
    return qvalues


def _flag(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0).eq(1)


def _verdict(rd: float, low: float, high: float) -> str:
    if math.isnan(rd):
        return "not_estimable"
    if rd > 0 and low > 0:
        return "lower_accuracy_significant"
    if rd < 0 and high < 0:
        return "higher_accuracy_significant"
    if rd > 0:
        return "lower_accuracy_inconclusive"
    if rd < 0:
        return "higher_accuracy_inconclusive"
    return "no_difference"


def _contrast(name: str, family: str, subgroup: pd.Series, reference: pd.Series,
              error: pd.Series, level: float) -> dict:
    n1 = int(subgroup.sum())
    n0 = int(reference.sum())
    e1 = int((subgroup & error).sum())
    e0 = int((reference & error).sum())
    rd, low, high = newcombe_risk_difference(e1, n1, e0, n0, level)
    pvalue = fisher_exact_two_sided(e1, n1 - e1, e0, n0 - e0)
    return {
        "subgroup": name,
        "family": family,
        "n_subgroup": n1,
        "n_reference": n0,
        "errors_subgroup": e1,
        "errors_reference": e0,
        "rate_subgroup": (e1 / n1) if n1 else math.nan,
        "rate_reference": (e0 / n0) if n0 else math.nan,
        "risk_difference": rd,
        "rd_ci_low": low,
        "rd_ci_high": high,
        "rd_ci_method": "newcombe_hybrid_score",
        "fisher_p": pvalue,
        "verdict": _verdict(rd, low, high),
        "ci_level": level,
    }


def _reference_masks(frame: pd.DataFrame, subgroup_columns: list[str],
                     reference_column: str | None) -> tuple[pd.Series, list[str]]:
    flags = {column: _flag(frame[column]) for column in subgroup_columns}
    if reference_column and reference_column in flags:
        reference = flags[reference_column]
        tested = [c for c in subgroup_columns if c != reference_column]
    else:
        # No declared reference: the reference group is "no flag set".
        reference = pd.Series(True, index=frame.index)
        for value in flags.values():
            reference &= ~value
        tested = list(subgroup_columns)
    return reference, tested


def subgroup_tests(frame: pd.DataFrame, reference_columns: list[str],
                   prediction: str, subgroup_columns: list[str],
                   subgroup_reference: str | None,
                   composites: dict[str, list[str]],
                   level: float) -> pd.DataFrame:
    """One row per (reference, subgroup) with the hypothesis contrast."""
    if not subgroup_columns:
        return pd.DataFrame()
    reference_mask, tested = _reference_masks(frame, subgroup_columns,
                                              subgroup_reference)
    rows = []
    for label_column in reference_columns:
        error = frame[label_column] != frame[prediction]
        block = []
        for name in tested:
            row = _contrast(name, "pre_specified", _flag(frame[name]),
                            reference_mask, error, level)
            block.append(row)
        pre_specified_p = [row["fisher_p"] for row in block]
        qvalues = benjamini_hochberg(pre_specified_p)
        for row, qvalue in zip(block, qvalues):
            row["bh_q"] = qvalue
            row["reference"] = label_column
            rows.append(row)
        for name, members in (composites or {}).items():
            present = [m for m in members if m in frame.columns]
            if not present:
                continue
            composite = pd.Series(False, index=frame.index)
            for member in present:
                composite |= _flag(frame[member])
            row = _contrast(name, "composite", composite, reference_mask,
                            error, level)
            row["bh_q"] = math.nan
            row["reference"] = label_column
            rows.append(row)
    columns = ["reference", "subgroup", "family", "n_subgroup", "n_reference",
               "errors_subgroup", "errors_reference", "rate_subgroup",
               "rate_reference", "risk_difference", "rd_ci_low", "rd_ci_high",
               "rd_ci_method", "fisher_p", "bh_q", "verdict", "ci_level"]
    return pd.DataFrame(rows, columns=columns)


def three_way_agreement(frame: pd.DataFrame, primary: str, extras: list[str],
                        prediction: str, level: float) -> pd.DataFrame:
    """AI vs primary vs extra-reference agreement decomposition."""
    rows = []
    for extra in extras:
        if extra not in frame.columns:
            continue
        primary_extra_agree = frame[primary] == frame[extra]
        ai_matches_primary = frame[prediction] == frame[primary]
        ai_matches_extra = frame[prediction] == frame[extra]
        categories = [
            ("all_three_agree", primary_extra_agree & ai_matches_primary),
            ("consensus_ai_disagrees", primary_extra_agree & ~ai_matches_primary),
            ("dispute_ai_sides_primary", ~primary_extra_agree & ai_matches_primary),
            ("dispute_ai_sides_extra", ~primary_extra_agree & ai_matches_extra),
            ("dispute_ai_sides_neither",
             ~primary_extra_agree & ~ai_matches_primary & ~ai_matches_extra),
        ]
        n = len(frame)
        for name, mask in categories:
            count = int(mask.sum())
            low, high = wilson(count, n, level)
            rows.append({
                "comparison": f"{primary}_vs_{extra}",
                "category": name,
                "n": count,
                "share": (count / n) if n else math.nan,
                "ci_low": low,
                "ci_high": high,
                "ci_method": "wilson",
                "ci_level": level,
            })
    columns = ["comparison", "category", "n", "share", "ci_low", "ci_high",
               "ci_method", "ci_level"]
    return pd.DataFrame(rows, columns=columns)


def reference_disputes(frame: pd.DataFrame, primary: str, extras: list[str],
                       subgroup_columns: list[str],
                       subgroup_reference: str | None,
                       composites: dict[str, list[str]],
                       level: float) -> pd.DataFrame:
    """How often the primary and secondary references disagree, by subgroup."""
    if not subgroup_columns or not extras:
        return pd.DataFrame()
    reference_mask, tested = _reference_masks(frame, subgroup_columns,
                                              subgroup_reference)
    rows = []
    for extra in extras:
        if extra not in frame.columns:
            continue
        dispute = frame[primary] != frame[extra]
        block = []
        for name in tested:
            row = _contrast(name, "pre_specified", _flag(frame[name]),
                            reference_mask, dispute, level)
            block.append(row)
        qvalues = benjamini_hochberg([row["fisher_p"] for row in block])
        for row, qvalue in zip(block, qvalues):
            row["bh_q"] = qvalue
            row["comparison"] = f"{primary}_vs_{extra}"
            rows.append(row)
        for name, members in (composites or {}).items():
            present = [m for m in members if m in frame.columns]
            if not present:
                continue
            composite = pd.Series(False, index=frame.index)
            for member in present:
                composite |= _flag(frame[member])
            row = _contrast(name, "composite", composite, reference_mask,
                            dispute, level)
            row["bh_q"] = math.nan
            row["comparison"] = f"{primary}_vs_{extra}"
            rows.append(row)
    columns = ["comparison", "subgroup", "family", "n_subgroup", "n_reference",
               "errors_subgroup", "errors_reference", "rate_subgroup",
               "rate_reference", "risk_difference", "rd_ci_low", "rd_ci_high",
               "rd_ci_method", "fisher_p", "bh_q", "verdict", "ci_level"]
    table = pd.DataFrame(rows, columns=columns)
    return table.rename(columns={"errors_subgroup": "disputes_subgroup",
                                 "errors_reference": "disputes_reference"})


# --------------------------------------------------------------------------- #
# Shared reading of the subgroup tables, used by both report composers.        #
# --------------------------------------------------------------------------- #

VERDICT_READING = {
    "lower_accuracy_significant":
        "significantly lower accuracy for the subgroup, supporting the hypothesis",
    "higher_accuracy_significant":
        "significantly higher accuracy for the subgroup, contradicting the hypothesis",
    "lower_accuracy_inconclusive":
        "lower accuracy in the hypothesised direction, but inconclusive "
        "(the interval includes zero)",
    "higher_accuracy_inconclusive":
        "higher accuracy, against the hypothesised direction, but inconclusive "
        "(the interval includes zero)",
    "no_difference": "no difference between the subgroup and the reference",
    "not_estimable": "not estimable",
}

_SIGNIFICANT_VERDICTS = ("lower_accuracy_significant", "higher_accuracy_significant")


def verdict_reading(verdict: str, bh_q: float | None = None,
                    family: str | None = None) -> str:
    """A reading of a subgroup verdict that respects multiplicity.

    A verdict is labelled *significant* when its risk-difference interval
    excludes zero. For a pre-specified row with a finite Benjamini-Hochberg q,
    the reading must not claim significance at 0.05 once the family-wise
    adjustment is applied: at ``q >= 0.05`` it is reported as not significant
    after correction. A composite row carries no q and is reported without a
    correction claim. When the correction still clears 0.05 the base reading is
    kept; the q value itself is shown in the table's ``BH q`` column.
    """
    base = VERDICT_READING.get(verdict, str(verdict))
    if verdict not in _SIGNIFICANT_VERDICTS:
        return base
    try:
        qvalue = None if bh_q is None else float(bh_q)
    except (TypeError, ValueError):
        qvalue = None
    if qvalue is None or math.isnan(qvalue) or qvalue < 0.05:
        return base
    direction = "lower" if verdict == "lower_accuracy_significant" else "higher"
    return (f"{direction} accuracy for the subgroup with an interval excluding "
            "zero, but not significant after Benjamini-Hochberg correction "
            f"(q = {qvalue:.3f})")


def reference_name(cfg: dict, column: str, short: bool = False) -> str:
    """Name a reference precisely.

    A study may declare a human-readable name for each reference column
    (``benchmark.reference_labels``) so that a star-derived proxy label is never
    described as a second human annotator. Without one, fall back to the
    primary/secondary role plus the column name.
    """
    labels = (cfg.get("benchmark", {}) or {}).get("reference_labels") or {}
    if column in labels:
        label = str(labels[column])
        return label if short else f"the {label}"
    primary = cfg["benchmark"]["label_column"]
    role = "primary" if column == primary else "secondary"
    if short:
        return f"{role} (`{column}`)"
    return f"the {role} reference (`{column}`)"


def hypothesis_summary(cfg: dict, results_dir: str | Path) -> str | None:
    """One sentence with the hypothesis verdict, or None if not applicable.

    The plan's pre-specified reference (``report.hypothesis_reference``) is
    reported first so the headline verdict answers the plan's own question.
    """
    path = Path(results_dir) / "t14_subgroup_tests.csv"
    if not path.exists():
        return None
    tests = pd.read_csv(path)
    if tests.empty:
        return None
    group = (cfg.get("report", {}).get("hypothesis_group") or "").strip()
    if not group:
        return None
    matched = tests[tests["subgroup"] == group]
    if matched.empty:
        return None
    plan_reference = (cfg.get("report", {}).get("hypothesis_reference")
                      or "").strip()
    if plan_reference and plan_reference in set(matched["reference"]):
        ordered = pd.concat([matched[matched["reference"] == plan_reference],
                             matched[matched["reference"] != plan_reference]])
    else:
        ordered = matched
    sentences = []
    for _, entry in ordered.iterrows():
        reading = verdict_reading(entry["verdict"], entry.get("bh_q"),
                                  entry.get("family"))
        sentences.append(
            f"against {reference_name(cfg, entry['reference'])} the hypothesis is "
            f"{reading} (risk difference "
            f"{100 * float(entry['risk_difference']):+.1f} pp, 95% CI "
            f"{100 * float(entry['rd_ci_low']):+.1f} to "
            f"{100 * float(entry['rd_ci_high']):+.1f}, Fisher p = "
            f"{float(entry['fisher_p']):.3f})")
    prefix = ""
    if plan_reference:
        prefix = (f"The plan's hypothesis is tested against "
                  f"{reference_name(cfg, plan_reference)}. ")
    return (prefix + f"Verdict on the pre-specified hypothesis (`{group}`): "
            + "; ".join(sentences) + ".")


def three_way_summary(cfg: dict, results_dir: str | Path) -> str | None:
    """The three-way split in plain words, or None when no secondary reference."""
    path = Path(results_dir) / "t15_three_way_agreement.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    comparison = str(frame.iloc[0]["comparison"])
    primary_column, _, secondary_column = comparison.partition("_vs_")
    primary = reference_name(cfg, primary_column, short=True)
    secondary = reference_name(cfg, secondary_column, short=True)
    counts = dict(zip(frame["category"], frame["n"]))
    consensus_ai = int(counts.get("consensus_ai_disagrees", 0))
    sides_secondary = int(counts.get("dispute_ai_sides_extra", 0))
    neither = int(counts.get("dispute_ai_sides_neither", 0))
    disagreements = consensus_ai + sides_secondary + neither
    noun = "disagreement" if disagreements == 1 else "disagreements"
    verb = "lies" if consensus_ai == 1 else "lie"
    return (f"Three-way agreement between the AI prediction, {primary} and "
            f"{secondary}: of the {disagreements} {noun} with {primary}, only "
            f"{consensus_ai} {verb} on rows where the two references agree; the "
            f"rest sit between them ({sides_secondary} side with {secondary}, "
            f"{neither} with neither).")

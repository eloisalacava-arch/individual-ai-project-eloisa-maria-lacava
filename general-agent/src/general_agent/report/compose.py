"""Compose one unique Markdown report for a single study.

The report combines the study's statistical tables (via the validated readers in
``evaluate.compare``) with the qualitative findings when stage 08 has run. If the
qualitative analysis has not run, the report says so and includes no explanation;
it never invents one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from ..evaluate.compare import collect_study, fmt
from ..evaluate.subgroups import (
    hypothesis_summary, reference_name, verdict_reading,
)
from ..limitations import dedupe_limitations
from ..provenance import file_sha256, now_iso
from ..qualitative.schema import reason_label, reason_label_lower
from . import paths

GENERIC_LIMITATIONS = [
    "The reference is a benchmark label, not an adjudicated gold standard; a "
    "disagreement is not a confirmed model error.",
    "Accuracy and recall use Wilson intervals; kappa, balanced accuracy, "
    "precision and F1 use a seeded percentile bootstrap.",
    "All intervals are conditional on this sample; the sample is not stratified.",
    "No causal claim is made. Statements are descriptive associations within one "
    "sample.",
    "Temperature is not controlled and model versions drift.",
]

RESULT_SOURCES = [
    "t01_join_and_sample.csv", "t02_accuracy.csv", "t03_kappa.csv",
    "t04_confusion_matrix.csv", "t05_per_class_metrics.csv",
    "t06_class_distribution.csv", "t07_ordinal_errors.csv",
    "t08_disagreements.csv", "t09_category_summary.csv", "t10_error_status.csv",
    "t11_disagreement_pairs.csv", "t12_text_profile.csv",
    "t12b_length_test.csv", "t13_reference_agreement.csv",
    "t14_subgroup_tests.csv", "t15_three_way_agreement.csv",
    "t16_reference_dispute.csv",
]


class ReportError(RuntimeError):
    """Raised when a report cannot be composed from the available files."""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _read_result(cfg: dict, name: str) -> pd.DataFrame:
    path = Path(cfg["_meta"]["output_dir"]) / "results" / name
    if not path.exists():
        raise ReportError(f"{path} is missing; run stages 05-06 for this study")
    return pd.read_csv(path)


def _pct(value, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not estimable"
    if math.isnan(number):
        return "not estimable"
    return f"{number * 100:.{digits}f}%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _source(name: str) -> str:
    return f"Source: `results/{name}`"


def _sentence(text: str) -> str:
    """A value with exactly one terminal punctuation mark.

    Reference and task descriptions often already end in a period; appending
    another would duplicate the punctuation in the report.
    """
    value = str(text or "").strip()
    if not value or value[-1] in ".!?":
        return value
    return value + "."


def _portable_config(cfg: dict) -> str:
    try:
        return str(Path(cfg["_meta"]["config_path"]).resolve().relative_to(paths.PROJECT_ROOT))
    except ValueError:
        return ""


def _recall_spread(classes: pd.DataFrame | None) -> dict | None:
    """Lowest and highest per-category recall, and whether they differ.

    A weakest category is only meaningful when one category is actually recalled
    less than another; identical recall (for example all 1.000) has no weakest
    category and must not be described as if it did.
    """
    if classes is None or len(classes) == 0:
        return None
    lowest = classes.loc[classes["recall"].idxmin()]
    highest = classes.loc[classes["recall"].idxmax()]
    return {
        "lowest": lowest,
        "highest": highest,
        "min": float(lowest["recall"]),
        "max": float(highest["recall"]),
        "differs": (float(highest["recall"]) - float(lowest["recall"])) > 1e-9,
    }


# --------------------------------------------------------------------------- #
# Config resolution                                                           #
# --------------------------------------------------------------------------- #

def resolve_study_config(run_id: str) -> Path | None:
    """Find a study config: in the run directory, else by run_id in ``configs/``."""
    candidate = paths.PROJECT_ROOT / "runs" / run_id / "study.yaml"
    if candidate.exists():
        return candidate
    if paths.CONFIG_DIR.exists():
        import yaml
        for path in sorted(paths.CONFIG_DIR.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (yaml.YAMLError, OSError):
                continue
            if isinstance(raw, dict) and (raw.get("run") or {}).get("run_id") == run_id:
                return path
    return None


def qualitative_available(cfg: dict) -> bool:
    return (Path(cfg["_meta"]["output_dir"]) / "qualitative"
            / "explanations.csv").exists()


# --------------------------------------------------------------------------- #
# Qualitative parts                                                           #
# --------------------------------------------------------------------------- #

def _qualitative_parts(cfg: dict) -> dict | None:
    from ..qualitative import narrative, patterns
    from ..qualitative.settings import load_settings

    run_dir = Path(cfg["_meta"]["output_dir"])
    out_dir = run_dir / "qualitative"
    if not (out_dir / "explanations.csv").exists():
        return None
    sidecar = out_dir / "settings.yaml"
    settings = load_settings(cfg["_meta"]["config_path"],
                             sidecar if sidecar.exists() else None)
    tables = out_dir / "tables"
    if not (tables / "q02_patterns.csv").exists():
        patterns.write_tables(out_dir)
    return {
        "settings": settings,
        "coverage": pd.read_csv(tables / "q05_coverage.csv"),
        "patterns": pd.read_csv(tables / "q02_patterns.csv"),
        "contrast": pd.read_csv(tables / "q07_reason_contrast.csv"),
        "blocks": narrative.qualitative_blocks(out_dir, settings),
    }


# --------------------------------------------------------------------------- #
# Sections                                                                    #
# --------------------------------------------------------------------------- #

def _metadata_table(cfg: dict, row: dict, qualitative: bool) -> list[str]:
    if qualitative:
        qualitative_value = "included (hypotheses only)"
    elif int(row["n_disagreements"]) == 0:
        qualitative_value = "skipped (no disagreements)"
    else:
        qualitative_value = "not run"
    return _table(["Field", "Value"], [
        ["Run id", f"`{cfg['run']['run_id']}`"],
        ["Generated", now_iso()],
        ["Model", f"`{cfg['model']['name']}`"],
        ["Seed", str(cfg["run"]["seed"])],
        ["Analysed rows", str(row["n"])],
        ["Categories", row["classes"]],
        ["Qualitative analysis", qualitative_value],
        ["Reference", _sentence(row["reference"])],
    ])


def _executive_summary(cfg: dict, row: dict, classes: pd.DataFrame,
                       qualitative: bool, parts: dict | None,
                       verdict: str | None = None) -> list[str]:
    lines: list[str] = []
    lines.append("#### Executive summary")
    lines.append("")
    lines.append(f"- {row['n']} observations were classified into "
                 f"{row['n_classes']} categories against the human reference.")
    lines.append(f"- Agreement: accuracy {fmt(row['accuracy'])} "
                 f"(95% CI {fmt(row['accuracy_ci_low'])}–{fmt(row['accuracy_ci_high'])}), "
                 f"balanced accuracy {fmt(row['balanced_accuracy'])}, Cohen's kappa "
                 f"{fmt(row['kappa_unweighted'])} "
                 f"(95% CI {fmt(row['kappa_ci_low'])}–{fmt(row['kappa_ci_high'])}).")
    lines.append(f"- {row['n_disagreements']} disagreements; the majority class "
                 f"`{row['majority_class']}` is {_pct(row['majority_share'])} of the "
                 "sample.")
    spread = _recall_spread(classes)
    if spread is not None:
        if spread["differs"]:
            lines.append(f"- Weakest category: `{spread['lowest']['class']}` "
                         f"(recall {fmt(spread['min'])} on "
                         f"{int(spread['lowest']['n_reference'])} reference rows).")
        else:
            lines.append(f"- Every category had the same recall "
                         f"({fmt(spread['min'])}); no weakest category to name.")
    if qualitative and parts is not None and len(parts["contrast"]):
        top = parts["contrast"].iloc[0]
        lines.append(f"- Qualitative hypotheses exist for "
                     f"{int(parts['coverage']['n_explained'].sum())} sampled items; the "
                     f"hypothesis most concentrated in disagreements is "
                     f"{reason_label_lower(top['reason'])} "
                     f"({_pct(top['difference_disagreements_minus_control'])} above "
                     "the control).")
    elif qualitative:
        lines.append("- Qualitative hypotheses are available for this study; the "
                     "patterns are summarised below.")
    elif int(row["n_disagreements"]) == 0:
        lines.append("- No disagreements were found, so qualitative analysis was "
                     "skipped: there was nothing to explain.")
    else:
        lines.append("- Qualitative analysis has not been run; none is included.")
    if verdict:
        lines.append(f"- {verdict}")
    lines.append("")
    if qualitative:
        lines.append("> Explanations are AI-generated hypotheses, not confirmed "
                     "causes. The reference is a benchmark, not adjudicated truth, "
                     "and no causal claim is made.")
    else:
        lines.append("> The reference is a benchmark, not adjudicated truth, and no "
                     "causal claim is made.")
    lines.append("")
    return lines


def _study_description(cfg: dict, row: dict) -> list[str]:
    task = cfg.get("task", {})
    lines: list[str] = []
    lines.append(f"- Task: {_sentence(task.get('role', 'not described'))}")
    if task.get("decision_rule"):
        lines.append(f"- Decision rule: {_sentence(task['decision_rule'])}")
    lines.append(f"- Reference: {_sentence(row['reference'])}")
    rationale = (cfg.get("report", {}).get("primary_reference_rationale")
                 or "").strip()
    if rationale:
        lines.append(f"- Why this primary reference: {_sentence(rationale)}")
    lines.append(f"- Categories and definitions:")
    for entry in cfg["labels"]["classes"]:
        lines.append(f"  - `{entry['name']}`: {entry['definition']}")
    lines.append(f"- Ordered categories: {bool(cfg['labels'].get('ordered'))}.")
    sample = cfg.get("sample", {})
    lines.append(f"- Sampling: seeded, unstratified (seed {cfg['run']['seed']}, "
                 f"n={sample.get('n')}, unique_text={sample.get('unique_text')}).")
    lines.append("- Predictions were frozen before the reference was opened "
                 "(stage 05); the scored copy is `results/frozen_predictions.csv`.")
    lines.append("")
    return lines


def _agreement(row: dict) -> list[str]:
    lines: list[str] = []
    lines.extend(_table(["Metric", "Value", "95% CI", "Method"], [
        ["Accuracy", fmt(row["accuracy"]),
         f"{fmt(row['accuracy_ci_low'])}–{fmt(row['accuracy_ci_high'])}", "Wilson"],
        ["Balanced accuracy", fmt(row["balanced_accuracy"]),
         f"{fmt(row['ba_ci_low'])}–{fmt(row['ba_ci_high'])}", "bootstrap"],
        ["Cohen's kappa (unweighted)", fmt(row["kappa_unweighted"]),
         f"{fmt(row['kappa_ci_low'])}–{fmt(row['kappa_ci_high'])}", "bootstrap"],
        ["Cohen's kappa (quadratic)",
         fmt(row["kappa_quadratic"]) if not math.isnan(row["kappa_quadratic"])
         else "not requested",
         (f"{fmt(row['kappa_q_ci_low'])}–{fmt(row['kappa_q_ci_high'])}"
          if not math.isnan(row["kappa_quadratic"]) else ""), "bootstrap"],
    ]))
    lines.append("")
    lines.append("Accuracy counts exact matches; kappa corrects for chance and "
                 "depends on the marginal distribution; balanced accuracy averages "
                 "per-class recall.")
    lines.append("")
    lines.append(_source("t02_accuracy.csv, t03_kappa.csv"))
    lines.append("")
    return lines


def _confusion(cfg: dict) -> list[str]:
    matrix = pd.read_csv(Path(cfg["_meta"]["output_dir"]) / "results"
                         / "t04_confusion_matrix.csv", index_col=0)
    headers = ["Human label \\ AI answer"] + [str(column) for column in matrix.columns]
    rows = [[str(index)] + [str(int(value)) for value in matrix.loc[index]]
            for index in matrix.index]
    lines = _table(headers, rows)
    lines.append("")
    lines.append("Rows are the human label; columns are the AI answer. The final "
                 "row and column are marginals.")
    lines.append("")
    lines.append(_source("t04_confusion_matrix.csv"))
    lines.append("")
    lines.append("<!-- figure: confusion_matrix | Confusion matrix, rows = human, "
                 "columns = AI -->")
    lines.append("")
    return lines


def _per_class(classes: pd.DataFrame) -> list[str]:
    rows = []
    for _, entry in classes.iterrows():
        rows.append([
            entry["class"], str(int(entry["n_reference"])),
            str(int(entry["n_predicted"])),
            fmt(entry["precision"]),
            f"{fmt(entry['recall'])} ({fmt(entry['recall_ci_low'])}–"
            f"{fmt(entry['recall_ci_high'])})",
            f"{fmt(entry['f1'])} ({fmt(entry['f1_ci_low'])}–{fmt(entry['f1_ci_high'])})",
        ])
    lines = _table(["Class", "Reference n", "Predicted n", "Precision",
                    "Recall (95% CI)", "F1 (95% CI)"], rows)
    lines.append("")
    lines.append(_source("t05_per_class_metrics.csv"))
    lines.append("")
    lines.append("<!-- figure: per_class_recall | Per-category recall with 95% "
                 "confidence interval -->")
    lines.append("")
    return lines


def _distribution(frame: pd.DataFrame) -> list[str]:
    rows = [[entry["class"], str(int(entry["human_n"])), _pct(entry["human_share"]),
             str(int(entry["ai_n"])), _pct(entry["ai_share"]),
             _pct(entry["difference_ai_minus_human"])]
            for _, entry in frame.iterrows()]
    lines = _table(["Class", "Human n", "Human share", "AI n", "AI share",
                    "Difference (AI minus human)"], rows)
    lines.append("")
    lines.append(_source("t06_class_distribution.csv"))
    lines.append("")
    return lines


def _disagreement_structure(severity: pd.DataFrame, pairs: pd.DataFrame,
                            n_disagreements: int, ordered: bool,
                            n_classes: int) -> list[str]:
    lines: list[str] = []
    lines.append(f"Total disagreements: **{n_disagreements}**.")
    lines.append("")
    rows = [[entry["bucket"].capitalize(), str(int(entry["n"])), _pct(entry["share"])]
            for _, entry in severity.iterrows()]
    lines.extend(_table(["Bucket", "n", "Share"], rows))
    lines.append("")
    total_rows = int(severity["n"].sum())
    if n_disagreements > 0:
        lines.append(f"The `Share` column is each bucket's share of all {total_rows} "
                     f"analysed rows, not of the {n_disagreements} disagreements; "
                     "the buckets sum to 100%.")
    else:
        lines.append(f"The `Share` column is each bucket's share of all {total_rows} "
                     "analysed rows; the buckets sum to 100%.")
    lines.append("")
    if ordered:
        lines.append("Adjacency is defined because the categories are ordered; the "
                     "`Adjacent` share is a severity proxy.")
    elif n_classes == 2:
        lines.append("With two categories every disagreement is at maximum distance, "
                     "so `Adjacent` is always zero.")
    else:
        lines.append("The categories are unordered, so there is no severity scale. "
                     "The distance buckets are computed from the declaration order "
                     "only and should not be read as adjacent or opposite.")
    lines.append("")
    top = pairs.sort_values("n", ascending=False).head(8)
    if not top.empty:
        rows = [[entry["reference"], entry["predicted"], str(int(entry["n"])),
                 _pct(entry["share_of_disagreements"])]
                for _, entry in top.iterrows()]
        lines.append("Most frequent disagreement directions:")
        lines.append("")
        lines.extend(_table(["Reference", "Prediction", "n",
                             "Share of disagreements"], rows))
        lines.append("")
    lines.append(_source("t07_ordinal_errors.csv, t11_disagreement_pairs.csv"))
    lines.append("")
    return lines


def _observation_length(text: dict) -> list[str]:
    lines = _table(["Group", "Median words"], [
        ["Agreement", fmt(text["median_words_agree"], 1)],
        ["Disagreement", fmt(text["median_words_disagree"], 1)],
    ])
    lines.append("")
    lines.append(f"Seeded permutation test p = {fmt(text['length_p'])}; one "
                 "pre-specified feature, no multiplicity adjustment (exploratory).")
    lines.append("")
    lines.append(_source("t12_text_profile.csv, t12b_length_test.csv"))
    lines.append("")
    return lines


def _secondary_references(references: pd.DataFrame) -> list[str]:
    if references.empty:
        return []
    rows = [[entry["comparison"], str(int(entry["n"])), fmt(entry["agreement"]),
             fmt(entry["kappa_unweighted"])]
            for _, entry in references.iterrows()]
    lines = _table(["Comparison", "n", "Agreement", "Kappa"], rows)
    lines.append("")
    lines.append("Secondary-reference rows describe how references relate to each "
                 "other and to the model; they do not confirm a model error.")
    lines.append("")
    lines.append(_source("t13_reference_agreement.csv"))
    lines.append("")
    return lines


def _read_optional(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


def _subgroup_section(cfg: dict, results_dir: Path) -> tuple[list[str], str | None]:
    """The pre-specified subgroup hypothesis test, when the study has one."""
    tests = _read_optional(results_dir, "t14_subgroup_tests.csv")
    if tests is None:
        return [], None
    lines: list[str] = []
    hypothesis = (cfg.get("report", {}).get("hypothesis") or "").strip()
    if hypothesis:
        lines.append(f"Pre-specified hypothesis: {hypothesis}")
        lines.append("")
    lines.append("Risk difference is the subgroup error rate minus the reference "
                 "rate, so a positive value means lower accuracy for the subgroup. "
                 "The interval is a Newcombe hybrid score; q is Benjamini-Hochberg "
                 "across the pre-specified flags only, so composite rows carry no q. "
                 "This is an observational association on one sample, not a causal "
                 "effect.")
    lines.append("")
    headers = ["Reference", "Subgroup", "Family", "Subgroup n/err",
               "Reference n/err", "Rate (subgroup)", "Rate (reference)",
               "Risk diff (pp)", "95% CI (pp)", "Fisher p", "BH q", "Reading"]
    rows = []
    for _, entry in tests.iterrows():
        qvalue = float(entry["bh_q"])
        rows.append([
            reference_name(cfg, entry["reference"], short=True), entry["subgroup"],
            entry["family"],
            f"{int(entry['n_subgroup'])}/{int(entry['errors_subgroup'])}",
            f"{int(entry['n_reference'])}/{int(entry['errors_reference'])}",
            _pct(entry["rate_subgroup"]), _pct(entry["rate_reference"]),
            f"{100 * float(entry['risk_difference']):+.1f}",
            f"{100 * float(entry['rd_ci_low']):+.1f} to "
            f"{100 * float(entry['rd_ci_high']):+.1f}",
            fmt(entry["fisher_p"]),
            "—" if math.isnan(qvalue) else fmt(qvalue),
            verdict_reading(entry["verdict"], qvalue, entry["family"]),
        ])
    lines.extend(_table(headers, rows))
    lines.append("")
    lines.append(_source("t14_subgroup_tests.csv"))
    lines.append("")

    disputes = _read_optional(results_dir, "t16_reference_dispute.csv")
    if disputes is not None:
        comparison = str(disputes.iloc[0]["comparison"])
        d_primary_column, _, d_secondary_column = comparison.partition("_vs_")
        d_primary = reference_name(cfg, d_primary_column, short=True)
        d_secondary = reference_name(cfg, d_secondary_column, short=True)
        lines.append(f"### Do {d_primary} and {d_secondary} disagree more on "
                     "difficult reviews?")
        lines.append("")
        lines.append(f"A dispute is a row where {d_primary} and {d_secondary} carry "
                     "different labels. Where the dispute rate is higher than the "
                     "straightforward reference, the two references are less settled, "
                     "so a model disagreement there is weaker evidence of a model "
                     "error.")
        lines.append("")
        d_headers = ["Comparison", "Subgroup", "Subgroup n/disputes",
                     "Reference n/disputes", "Rate (subgroup)",
                     "Rate (reference)", "Risk diff (pp)", "95% CI (pp)",
                     "Fisher p", "BH q"]
        d_rows = []
        for _, entry in disputes.iterrows():
            qvalue = float(entry["bh_q"])
            d_rows.append([
                entry["comparison"], entry["subgroup"],
                f"{int(entry['n_subgroup'])}/{int(entry['disputes_subgroup'])}",
                f"{int(entry['n_reference'])}/{int(entry['disputes_reference'])}",
                _pct(entry["rate_subgroup"]), _pct(entry["rate_reference"]),
                f"{100 * float(entry['risk_difference']):+.1f}",
                f"{100 * float(entry['rd_ci_low']):+.1f} to "
                f"{100 * float(entry['rd_ci_high']):+.1f}",
                fmt(entry["fisher_p"]),
                "—" if math.isnan(qvalue) else fmt(qvalue),
            ])
        lines.extend(_table(d_headers, d_rows))
        lines.append("")
        lines.append(_source("t16_reference_dispute.csv"))
        lines.append("")

    group = (cfg.get("report", {}).get("hypothesis_group") or "").strip()
    verdict = hypothesis_summary(cfg, results_dir) if group else None
    return lines, verdict


def _three_way_section(cfg: dict, results_dir: Path) -> list[str]:
    table = _read_optional(results_dir, "t15_three_way_agreement.csv")
    if table is None:
        return []
    comparison = str(table.iloc[0]["comparison"])
    primary_column, _, secondary_column = comparison.partition("_vs_")
    primary = reference_name(cfg, primary_column, short=True)
    secondary = reference_name(cfg, secondary_column, short=True)
    labels = {
        "all_three_agree": "All three agree",
        "consensus_ai_disagrees": f"{primary} and {secondary} agree; AI differs",
        "dispute_ai_sides_primary": f"References differ; AI agrees with {primary}",
        "dispute_ai_sides_extra": f"References differ; AI agrees with {secondary}",
        "dispute_ai_sides_neither": "References differ; AI agrees with neither",
    }
    lines = [f"This table decomposes every row by whether the AI prediction matches "
             f"the primary reference ({primary}) and the secondary reference "
             f"({secondary}). The `{labels['consensus_ai_disagrees']}` row is the "
             "only kind where the two references agree and the AI differs; on the "
             "`references differ` rows the two references already disagree, so a "
             "disagreement there is not established as a model error. A secondary "
             "reference may be a proxy label rather than a second human annotation."]
    lines.append("")
    headers = ["Comparison", "Category", "n", "Share", "95% CI"]
    rows = [[f"{primary} vs {secondary}",
             labels.get(entry["category"], entry["category"]),
             int(entry["n"]), _pct(entry["share"]),
             f"{_pct(entry['ci_low'])}-{_pct(entry['ci_high'])}"]
            for _, entry in table.iterrows()]
    lines.extend(_table(headers, rows))
    lines.append("")
    lines.append(_source("t15_three_way_agreement.csv"))
    lines.append("")
    return lines


def _qualitative_findings(parts: dict, row: dict) -> list[str]:
    lines: list[str] = []
    lines.append("Explanations below are AI-generated hypotheses, not confirmed "
                 "causes, and not proof that the classifier or the human made a "
                 "mistake.")
    lines.append("")
    coverage = parts["coverage"]
    selected = coverage.set_index("group")["n_selected"].to_dict()
    disagreements_selected = int(selected.get("disagreement", 0))
    agreements_selected = int(selected.get("agreement", 0))
    total_disagreements = int(row["n_disagreements"])
    total_agreements = int(row["n"]) - total_disagreements
    lines.append(
        f"Selection coverage: {disagreements_selected} of {total_disagreements} "
        f"disagreements and {agreements_selected} of {total_agreements} "
        "agreements were sampled for explanation.")
    lines.append("")
    rows = [[entry["group"].capitalize(), str(int(entry["n_selected"])),
             str(int(entry["n_explained"])), str(int(entry["n_failed"])),
             _pct(entry["explained_share"])]
            for _, entry in coverage.iterrows()]
    lines.extend(_table(["Group", "Selected", "Explained", "Failed",
                         "Explained share"], rows))
    lines.append("")
    lines.append("No multiplicity adjustment is applied to these reason contrasts; "
                 "they are exploratory hypotheses, and the control sample is small.")
    lines.append("")
    lines.extend(parts["blocks"]["recurring"])
    lines.append("### Illustrative examples")
    lines.append("")
    lines.extend(parts["blocks"]["examples"])
    lines.append(_source("qualitative/tables/q02_patterns.csv, q04_examples.csv, "
                         "q05_coverage.csv, q07_reason_contrast.csv"))
    lines.append("")
    return lines


def _qualitative_absent(row: dict) -> list[str]:
    if int(row["n_disagreements"]) == 0:
        return [
            "> Qualitative analysis was skipped: the model and the reference agreed "
            "on every row, so there were no disagreements to explain. No explanations "
            "are included, and none have been invented.",
            "",
        ]
    return [
        "> Qualitative analysis has not been run for this study. No explanations "
        "are included, and none have been invented. Run stage 08 "
        "(`scripts/08_qualitative.py`) to add hypotheses about disagreements.",
        "",
    ]


def _interpretation(row: dict, classes: pd.DataFrame, severity: pd.DataFrame,
                    distribution: pd.DataFrame, ordered: bool, n_classes: int,
                    qualitative: bool, parts: dict | None) -> list[str]:
    lines: list[str] = []
    lines.append(f"- The model agreed with the human reference on "
                 f"{fmt(row['accuracy'])} of {row['n']} analysed rows "
                 f"(kappa {fmt(row['kappa_unweighted'])}); balanced accuracy "
                 f"{fmt(row['balanced_accuracy'])} shows this is not purely a "
                 "dominant-class effect.")
    lines.append(f"- The majority reference class `{row['majority_class']}` is "
                 f"{_pct(row['majority_share'])} of the sample, which raises chance "
                 "agreement and makes kappa harder to compare across studies.")
    spread = _recall_spread(classes)
    if spread is not None:
        if spread["differs"]:
            lines.append(f"- Recall ranged from {fmt(spread['min'])} for "
                         f"`{spread['lowest']['class']}` to {fmt(spread['max'])} for "
                         f"`{spread['highest']['class']}`; the lower figure means that "
                         "category's reference rows were matched less often. It does "
                         "not say how many rows that is, or why.")
        else:
            lines.append(f"- Every category was recalled equally "
                         f"({fmt(spread['min'])}); no category stands out, so "
                         "per-category recall gives no signal here.")
    if not distribution.empty:
        biased = distribution.reindex(
            distribution["difference_ai_minus_human"].abs().sort_values(
                ascending=False).index).iloc[0]
        difference = float(biased["difference_ai_minus_human"])
        direction = "less" if difference < 0 else "more"
        lines.append(f"- The AI predicted `{biased['class']}` "
                     f"{abs(difference) * 100:.1f} percentage points {direction} "
                     "often than the human reference did.")
    if ordered:
        counts = severity.set_index("bucket")["n"].to_dict()
        shares = severity.set_index("bucket")["share"].to_dict()
        total_rows = int(severity["n"].sum())
        lines.append(
            f"- Of all {total_rows} analysed rows, "
            f"{_pct(shares.get('adjacent', 0))} "
            f"({int(counts.get('adjacent', 0))}) were adjacent disagreements and "
            f"{_pct(shares.get('opposite', 0))} "
            f"({int(counts.get('opposite', 0))}) were opposite; the adjacent share "
            "gauges how severe the errors are.")
    elif n_classes == 2:
        lines.append("- With two categories every disagreement is at maximum "
                     "distance, so severity cannot be graded.")
    else:
        lines.append("- The categories are unordered, so severity is not graded; the "
                     "distance buckets follow the declaration order only.")
    if qualitative and parts is not None and len(parts["contrast"]):
        contrast = parts["contrast"]
        top = contrast.iloc[0]
        lines.append(f"- Qualitatively, {reason_label_lower(top['reason'])} was the "
                     "hypothesis most concentrated in disagreements "
                     f"({_pct(top['difference_disagreements_minus_control'])} above "
                     "the control).")
        shared = contrast[contrast["difference_disagreements_minus_control"] <= 0]
        if len(shared):
            lines.append(f"- {reason_label(shared.iloc[0]['reason'])} appeared in "
                         "both disagreements and the control, so it is not specific "
                         "to disagreement.")
    lines.append("")
    return lines


def _conclusions(row: dict, classes: pd.DataFrame, qualitative: bool,
                 parts: dict | None, hypothesis: str | None = None) -> list[str]:
    lines: list[str] = [
        f"- On this sample the model agreed with the reference on "
        f"{fmt(row['accuracy'])} of rows, with kappa {fmt(row['kappa_unweighted'])}.",
    ]
    spread = _recall_spread(classes)
    if spread is not None and spread["differs"]:
        lines.append(f"- The category most often missed was "
                     f"`{spread['lowest']['class']}` (recall {fmt(spread['min'])}).")
    elif spread is not None:
        lines.append(f"- No category was missed more than another; every category's "
                     f"recall was {fmt(spread['min'])}.")
    if qualitative and parts is not None and len(parts["contrast"]):
        top = parts["contrast"].iloc[0]
        lines.append(f"- Qualitatively, {reason_label_lower(top['reason'])} was the "
                     "hypothesis most concentrated in disagreements; it, and every "
                     "other hypothesis here, remains unconfirmed.")
    elif qualitative:
        lines.append("- Qualitative explanations are hypotheses about possible "
                     "reasons for disagreement, not confirmed causes.")
    elif int(row["n_disagreements"]) == 0:
        lines.append("- No disagreements occurred, so qualitative analysis was "
                     "skipped; there was nothing to explain.")
    else:
        lines.append("- No qualitative explanation of disagreements has been run "
                     "for this study.")
    if hypothesis:
        lines.append(f"- {hypothesis}")
    lines.append("- These are descriptive associations within one sample; no causal "
                 "claim is made, and a disagreement is not a confirmed model error.")
    lines.append("")
    return lines


def _limitations(cfg: dict, row: dict, qualitative: bool,
                 parts: dict | None) -> list[str]:
    lines = list(GENERIC_LIMITATIONS)
    lines.extend(cfg.get("report", {}).get("limitations", []) or [])
    if qualitative and parts is not None:
        lines.extend(parts["blocks"]["limitations"])
    lines.append(f"{row['n']} analysed rows; every interval is conditional on "
                 "this sample.")
    return dedupe_limitations(lines) + [""]


def _reproducibility(cfg: dict, qualitative: bool) -> list[str]:
    run_id = cfg["run"]["run_id"]
    portable = _portable_config(cfg)
    config_arg = portable or "<config>"
    lines = ["```"]
    lines.append(f"uv run python scripts/05_evaluate.py --config {config_arg}")
    lines.append(f"uv run python scripts/06_disagreements.py --config {config_arg}")
    lines.append(f"uv run python scripts/07_compare.py --configs {config_arg}")
    if qualitative:
        lines.append(f"uv run python scripts/08_qualitative.py --config {config_arg} "
                     f"--settings runs/{run_id}/qualitative/settings.yaml --report")
    lines.append(f"uv run python scripts/09_report.py --run-id {run_id}")
    lines.append("```")
    lines.append("")
    lines.append(f"Config sha256 `{cfg['_meta']['config_sha256']}`; model "
                 f"`{cfg['model']['name']}`; seed {cfg['run']['seed']}.")
    lines.append("")
    lines.append("Every number in this report is formatted from a CSV under "
                 "`results/` (and `qualitative/tables/` when present).")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Composition                                                                 #
# --------------------------------------------------------------------------- #

def compose_study(config_path: str | Path) -> tuple[str, dict]:
    """Return ``(markdown, cfg)`` for one study. Reads tables only."""
    from ..config import load_config

    cfg = load_config(config_path)
    row, classes, severity, references, pairs, text = collect_study(cfg)
    distribution = _read_result(cfg, "t06_class_distribution.csv")
    parts = _qualitative_parts(cfg)
    qualitative = parts is not None
    ordered = bool(cfg["labels"].get("ordered"))
    results_dir = Path(cfg["_meta"]["output_dir"]) / "results"
    subgroup, verdict = _subgroup_section(cfg, results_dir)
    three_way = _three_way_section(cfg, results_dir)
    secondary = _secondary_references(references)

    lines: list[str] = []
    lines.append(f"# {row['title']} — study report")
    lines.append("")
    lines.extend(_metadata_table(cfg, row, qualitative))
    lines.append("")
    lines.extend(_executive_summary(cfg, row, classes, qualitative, parts,
                                    verdict))

    number = 0

    def section(title: str, body: list[str]) -> None:
        nonlocal number
        number += 1
        lines.append(f"## {number}. {title}")
        lines.append("")
        lines.extend(body)

    section("Study description", _study_description(cfg, row))
    section("Agreement with the reference", _agreement(row))
    section("Confusion matrix", _confusion(cfg))
    section("Per-category results", _per_class(classes))
    section("Class distribution", _distribution(distribution))
    section("Disagreement structure",
            _disagreement_structure(severity, pairs, row["n_disagreements"],
                                    ordered, row["n_classes"]))
    if subgroup:
        section("Subgroup hypothesis test", subgroup)
    if three_way:
        section("Three-way agreement", three_way)
    section("Observation length (exploratory)", _observation_length(text))
    if secondary:
        section("Secondary reference structure", secondary)
    if qualitative:
        section("Qualitative findings", _qualitative_findings(parts, row))
    else:
        section("Qualitative findings", _qualitative_absent(row))
    section("Interpretation",
            _interpretation(row, classes, severity, distribution, ordered,
                            row["n_classes"], qualitative, parts))
    section("Conclusions", _conclusions(row, classes, qualitative, parts, verdict))
    section("Limitations", _limitations(cfg, row, qualitative, parts))
    section("Reproducibility", _reproducibility(cfg, qualitative))

    return "\n".join(lines) + "\n", cfg


def write_study_report(config_path: str | Path,
                       out_dir: str | Path | None = None) -> Path:
    """Write ``study_report.md`` (and metadata) for one study. No PDF here."""
    markdown, cfg = compose_study(config_path)
    run_id = cfg["run"]["run_id"]
    directory = Path(out_dir) if out_dir else paths.report_dir(run_id)
    paths.ensure_dir(directory)
    target = directory / "study_report.md"
    target.write_text(markdown, encoding="utf-8")

    run_dir = Path(cfg["_meta"]["output_dir"])
    sources = {}
    for name in RESULT_SOURCES:
        path = run_dir / "results" / name
        if path.exists():
            sources[f"results/{name}"] = file_sha256(path)
    qualitative_explanations = run_dir / "qualitative" / "explanations.csv"
    if qualitative_explanations.exists():
        sources["qualitative/explanations.csv"] = file_sha256(qualitative_explanations)
    (directory / "report_meta.json").write_text(
        json.dumps({
            "run_id": run_id,
            "generated_at": now_iso(),
            "config_path": cfg["_meta"]["config_path"],
            "config_sha256": cfg["_meta"]["config_sha256"],
            "qualitative": qualitative_explanations.exists(),
            "sources": sources,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target

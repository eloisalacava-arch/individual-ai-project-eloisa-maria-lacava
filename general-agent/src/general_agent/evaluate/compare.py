"""Stage 07 - cross-dataset comparison and generated report.

Inputs are the per-run result tables only. The same code reads any number of
configurations; there is no dataset-specific branch anywhere in this module.

Outputs, under ``comparison/``:
    tables/cross_dataset_metrics.csv
    tables/cross_dataset_classes.csv
    tables/cross_dataset_severity.csv
    tables/cross_dataset_references.csv
    figures/fig1_accuracy_kappa.svg
    figures/fig2_class_recall.svg
    figures/fig3_severity.svg
    report.md
    comparison_log.txt
    run_metadata.json

Every number in the report is formatted from one of those tables at generation
time; no value is typed into the prose.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .. import __version__
from ..config import load_config  # noqa: E402
from ..limitations import dedupe_limitations  # noqa: E402
from ..provenance import file_sha256, now_iso  # noqa: E402
from ..qualitative.schema import reason_label, reason_label_lower  # noqa: E402
from .subgroups import hypothesis_summary, three_way_summary  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# A per-study limitation line is prefixed with the run id, e.g. ``\`study_a\`: ...``.
_LIMITATION_PREFIX = re.compile(r"^`[^`]+`:\s*")

REQUIRED = ["t02_accuracy.csv", "t03_kappa.csv", "t05_per_class_metrics.csv",
            "t06_class_distribution.csv", "t07_ordinal_errors.csv",
            "t11_disagreement_pairs.csv"]


def _result(cfg: dict, name: str) -> pd.DataFrame:
    path = Path(cfg["_meta"]["output_dir"]) / "results" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run stages 05-06 for this config")
    return pd.read_csv(path)


def _metric(t03: pd.DataFrame, key: str, column: str) -> float:
    rows = t03[t03["metric"].str.contains(key)]
    if rows.empty:
        return math.nan
    return float(rows.iloc[0][column])


def collect_study(cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame, dict]:
    """Read one study's result tables. No statistic is recomputed here.

    Returns ``(summary_row, classes, severity, references, pairs, text)``, exactly
    the frames the comparison report and the per-study report format from. This is
    public so the report composer reuses the validated readers rather than
    re-deriving anything.
    """
    results = Path(cfg["_meta"]["output_dir"]) / "results"
    t02 = _result(cfg, "t02_accuracy.csv").iloc[0]
    t03 = _result(cfg, "t03_kappa.csv")
    t05 = _result(cfg, "t05_per_class_metrics.csv")
    t06 = _result(cfg, "t06_class_distribution.csv")
    t07 = _result(cfg, "t07_ordinal_errors.csv")
    t11 = _result(cfg, "t11_disagreement_pairs.csv")

    majority = t06.sort_values("human_n", ascending=False).iloc[0]
    n_classes = int(len(t05))
    # Missing report metadata is derived rather than left blank. The benchmark
    # file name is always available and is more useful than "not described".
    reference = (cfg.get("report", {}).get("reference_description")
                 or f"Human labels from `{Path(cfg['benchmark']['path']).name}`.")
    task = (cfg.get("report", {}).get("task_description")
            or f"{n_classes}-category classification.")
    row = {
        "dataset": cfg["run"]["run_id"],
        "title": (cfg.get("report", {}).get("title") or cfg["run"]["run_id"]),
        "task": task,
        "reference": reference,
        "n": int(t02["n"]),
        "n_classes": n_classes,
        "ordered": bool(cfg["labels"].get("ordered")),
        "classes": " / ".join(t05["class"].tolist()),
        "majority_class": majority["class"],
        "majority_share": float(majority["human_share"]),
        "accuracy": float(t02["accuracy"]),
        "accuracy_ci_low": float(t02["accuracy_ci_low"]),
        "accuracy_ci_high": float(t02["accuracy_ci_high"]),
        "balanced_accuracy": float(t02["balanced_accuracy"]),
        "ba_ci_low": float(t02["ba_ci_low"]),
        "ba_ci_high": float(t02["ba_ci_high"]),
        "kappa_unweighted": _metric(t03, "unweighted", "value"),
        "kappa_ci_low": _metric(t03, "unweighted", "ci_low"),
        "kappa_ci_high": _metric(t03, "unweighted", "ci_high"),
        "kappa_quadratic": _metric(t03, "quadratic", "value"),
        "kappa_q_ci_low": _metric(t03, "quadratic", "ci_low"),
        "kappa_q_ci_high": _metric(t03, "quadratic", "ci_high"),
        "seed": int(cfg["run"]["seed"]),
        "n_disagreements": int(t11.loc[t11["ordinal_distance"] > 0, "n"].sum()),
    }

    classes = t05.copy()
    classes.insert(0, "dataset", cfg["run"]["run_id"])
    severity = t07.copy()
    severity.insert(0, "dataset", cfg["run"]["run_id"])
    pairs = t11.copy()
    pairs = pairs[pairs["ordinal_distance"] > 0].copy()
    pairs.insert(0, "dataset", cfg["run"]["run_id"])

    references_path = results / "t13_reference_agreement.csv"
    if references_path.exists():
        references = pd.read_csv(references_path)
        references.insert(0, "dataset", cfg["run"]["run_id"])
    else:
        references = pd.DataFrame(columns=[
            "dataset", "comparison", "n", "agreement", "kappa_unweighted",
            "kappa_quadratic", "note"])

    text = {"dataset": cfg["run"]["run_id"],
            "median_words_agree": float("nan"),
            "median_words_disagree": float("nan"),
            "length_p": float("nan")}
    profile_path = results / "t12_text_profile.csv"
    test_path = results / "t12b_length_test.csv"
    if profile_path.exists():
        profile = pd.read_csv(profile_path).set_index("group")
        if "agree" in profile.index:
            text["median_words_agree"] = float(profile.loc["agree", "median_words"])
        if "disagree" in profile.index:
            text["median_words_disagree"] = float(profile.loc["disagree", "median_words"])
    if test_path.exists():
        text["length_p"] = float(pd.read_csv(test_path).iloc[0]["p_two_sided"])
    return row, classes, severity, references, pairs, text


def _figure_metrics(metrics: pd.DataFrame, path: Path) -> None:
    labels = metrics["dataset"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "Accuracy", axes[0]),
        ("kappa_unweighted", "kappa_ci_low", "kappa_ci_high", "Cohen's kappa", axes[1]),
    ]
    for column, low, high, title, axis in panels:
        values = metrics[column].to_numpy(float)
        errors = [values - metrics[low].to_numpy(float),
                  metrics[high].to_numpy(float) - values]
        axis.errorbar(values, range(len(labels)), xerr=errors, fmt="o", capsize=4)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_xlim(0, 1.0)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.3)
    fig.suptitle("Agreement with the human reference (95% CI)")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _figure_classes(classes: pd.DataFrame, path: Path) -> None:
    datasets = classes["dataset"].unique().tolist()
    fig, axis = plt.subplots(figsize=(10, 4.6))
    width = 0.8 / max(len(datasets), 1)
    for offset, dataset in enumerate(datasets):
        subset = classes[classes["dataset"] == dataset]
        positions = [index + offset * width for index in range(len(subset))]
        values = subset["recall"].to_numpy(float)
        errors = [values - subset["recall_ci_low"].to_numpy(float),
                  subset["recall_ci_high"].to_numpy(float) - values]
        axis.bar(positions, values, width=width, label=dataset, yerr=errors,
                 capsize=3, alpha=0.8)
        axis.set_xticks([index + width * (len(datasets) - 1) / 2
                         for index in range(len(subset))])
        axis.set_xticklabels(subset["class"])
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Recall (95% CI)")
    axis.set_title("Per-class recall by dataset")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _figure_severity(severity: pd.DataFrame, path: Path) -> None:
    buckets = ["agree", "adjacent", "opposite"]
    datasets = severity["dataset"].unique().tolist()
    fig, axis = plt.subplots(figsize=(9, 4.2))
    bottom = [0.0] * len(datasets)
    colours = {"agree": "#4c9f70", "adjacent": "#d9a441", "opposite": "#b5446e"}
    for bucket in buckets:
        shares = []
        for dataset in datasets:
            subset = severity[(severity["dataset"] == dataset)
                              & (severity["bucket"] == bucket)]
            shares.append(float(subset["share"].iloc[0]) if len(subset) else 0.0)
        axis.bar(datasets, shares, bottom=bottom, label=bucket, color=colours[bucket])
        bottom = [current + share for current, share in zip(bottom, shares)]
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Share of observations")
    axis.set_title("Ordinal distance from the reference label")
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def _fmt(value: float, digits: int = 3) -> str:
    return "not estimable" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"{value:.{digits}f}"


# Public alias so report composers format numbers the same way as stage 07.
fmt = _fmt


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


def _portable_path(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return ""


def _qualitative_summary(cfg: dict) -> dict | None:
    """Read one study's qualitative pattern tables, if they exist.

    Generic: it reads files by path only and never mentions a dataset name.
    """
    run_dir = Path(cfg["_meta"]["output_dir"])
    out_dir = run_dir / "qualitative"
    if not (out_dir / "explanations.csv").exists():
        return None
    tables = out_dir / "tables"
    if not (tables / "q02_patterns.csv").exists():
        return None
    summary: dict = {"run_id": cfg["run"]["run_id"]}
    coverage = pd.read_csv(tables / "q05_coverage.csv")
    summary["explained"] = int(coverage["n_explained"].sum())
    summary["selected"] = int(coverage["n_selected"].sum())
    by_group = coverage.set_index("group")["n_selected"].to_dict()
    summary["selected_disagreement"] = int(by_group.get("disagreement", 0))
    summary["selected_agreement"] = int(by_group.get("agreement", 0))
    patterns = pd.read_csv(tables / "q02_patterns.csv")
    contrast = pd.read_csv(tables / "q07_reason_contrast.csv")
    disagreements = patterns[patterns["group"] == "disagreement"].sort_values(
        "n", ascending=False)
    controls = patterns[patterns["group"] == "agreement"].sort_values(
        "n", ascending=False)
    summary["top_disagreement"] = (disagreements.iloc[0].to_dict()
                                   if len(disagreements) else None)
    summary["top_control"] = (controls.iloc[0].to_dict() if len(controls) else None)
    summary["contrast"] = contrast
    return summary


def _qualitative_section(qualitative: list[dict | None],
                         metrics: pd.DataFrame) -> list[str]:
    present = [entry for entry in qualitative if entry]
    lines = ["## 5. Qualitative findings (AI hypotheses)", ""]
    if not present:
        if len(metrics) and int(metrics["n_disagreements"].sum()) == 0:
            lines.append("Qualitative analysis was skipped for every study: there "
                         "were no disagreements to explain.")
        else:
            lines.append("No study has a qualitative analysis yet. Run stage 08 for a "
                         "study, then regenerate this comparison.")
        lines.append("")
        return lines
    lines.append("Explanations are AI-generated hypotheses, not confirmed causes, "
                 "and not proof of a model error. Each row reads that study's own "
                 "qualitative tables.")
    lines.append("")
    metrics_by_id = {row["dataset"]: row for _, row in metrics.iterrows()}
    for summary in present:
        lines.append(f"### `{summary['run_id']}`")
        lines.append("")
        row = metrics_by_id.get(summary["run_id"])
        if row is not None:
            total_disagreements = int(row["n_disagreements"])
            total_agreements = int(row["n"]) - total_disagreements
            lines.append(
                f"{summary['selected_disagreement']} of {total_disagreements} "
                f"disagreements and {summary['selected_agreement']} of "
                f"{total_agreements} agreements were sampled; "
                f"{summary['explained']} of {summary['selected']} were explained.")
        else:
            lines.append(f"{summary['explained']} of {summary['selected']} sampled "
                         "items were explained.")
        lines.append("")
        contrast = summary["contrast"].sort_values(
            "difference_disagreements_minus_control", ascending=False).head(5)
        rows = [[reason_label(entry["reason"]),
                 _pct(entry["share_disagreements"]),
                 _pct(entry["share_control"]),
                 _pct(entry["difference_disagreements_minus_control"])]
                for _, entry in contrast.iterrows()]
        lines.extend(_table(["Reason", "Disagreements", "Control", "Difference"],
                            rows))
        lines.append("")
    lines.append("A reason shared by both groups is not specific to disagreement; "
                 "each control is small and its explanations are hypotheses too. No "
                 "multiplicity adjustment is applied to these contrasts.")
    lines.append("")
    return lines


def _executive_summary(metrics: pd.DataFrame, qualitative: list[dict | None]) -> list[str]:
    study_ids = list(metrics["dataset"])
    lines = ["#### Executive summary", ""]
    lines.append(f"- This report compares {len(study_ids)} studies: "
                 + ", ".join(f"`{name}`" for name in study_ids) + ".")
    for _, row in metrics.iterrows():
        lines.append(
            f"- `{row['dataset']}`: n={int(row['n'])}, accuracy "
            f"{_fmt(row['accuracy'])} (95% CI {_fmt(row['accuracy_ci_low'])}–"
            f"{_fmt(row['accuracy_ci_high'])}), kappa {_fmt(row['kappa_unweighted'])}; "
            f"majority class `{row['majority_class']}` "
            f"({_pct(row['majority_share'])}).")
    by_id = {entry["run_id"]: entry for entry in qualitative if entry}
    explained = []
    for name in study_ids:
        entry = by_id.get(name)
        if entry and entry.get("top_disagreement"):
            explained.append(
                f"`{name}`: {reason_label_lower(entry['top_disagreement']['reason'])}")
        else:
            explained.append(f"`{name}`: not run")
    lines.append("- Qualitative top hypothesis per study — " + "; ".join(explained)
                 + ".")
    lines.append("")
    lines.append("> Differences below are differences between studies, not effects of "
                 "any single factor. References are benchmarks, not adjudicated "
                 "truth, and no causal claim is made.")
    lines.append("")
    return lines


def _report(metrics: pd.DataFrame, classes: pd.DataFrame, severity: pd.DataFrame,
            references: pd.DataFrame, pairs: pd.DataFrame, text: pd.DataFrame,
            configs: list[dict], path: Path,
            qualitative: list[dict | None] | None = None) -> None:
    qualitative = qualitative or []
    comparative = len(configs) > 1
    lines: list[str] = []
    scope = ("Cross-dataset comparison" if comparative else "Study report")
    lines.append(f"# {scope}: AI classification against a human reference")
    lines.append("")
    lines.append(f"Generated by `scripts/07_compare.py` (general-agent {__version__}) "
                 f"from the result tables and, where a study has one, its qualitative "
                 f"tables. {now_iso()}")
    lines.append("")
    if comparative:
        lines.extend(_executive_summary(metrics, qualitative))
    lines.append("")

    lines.append("## 1. What is being compared")
    lines.append("")
    lines.append("| | " + " | ".join(metrics["dataset"]) + " |")
    lines.append("|---|" + "---|" * len(metrics))
    for field, label in (
        ("task", "Task"), ("classes", "Classes"), ("reference", "Reference"),
        ("n", "Analysed rows"), ("majority_class", "Majority class"),
        ("majority_share", "Majority share"),
    ):
        values = []
        for _, row in metrics.iterrows():
            value = row[field]
            values.append(_pct(value) if field == "majority_share" else str(value))
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.append("")

    lines.append("## 2. Agreement with the reference")
    lines.append("")
    lines.append("| Dataset | Accuracy (95% CI) | Balanced accuracy (95% CI) | "
                 "Kappa (95% CI) | Quadratic kappa (95% CI) |")
    lines.append("|---|---|---|---|---|")
    for _, row in metrics.iterrows():
        quadratic = (_fmt(row["kappa_quadratic"]) if not math.isnan(row["kappa_quadratic"])
                     else "not requested")
        quadratic_ci = (f" ({_fmt(row['kappa_q_ci_low'])}-{_fmt(row['kappa_q_ci_high'])})"
                        if not math.isnan(row["kappa_quadratic"]) else "")
        lines.append(
            f"| {row['dataset']} | {_fmt(row['accuracy'])} "
            f"({_fmt(row['accuracy_ci_low'])}-{_fmt(row['accuracy_ci_high'])}) | "
            f"{_fmt(row['balanced_accuracy'])} "
            f"({_fmt(row['ba_ci_low'])}-{_fmt(row['ba_ci_high'])}) | "
            f"{_fmt(row['kappa_unweighted'])} "
            f"({_fmt(row['kappa_ci_low'])}-{_fmt(row['kappa_ci_high'])}) | "
            f"{quadratic}{quadratic_ci} |")
    lines.append("")
    lines.append("Accuracy counts exact matches. Cohen's kappa corrects for chance "
                 "agreement and depends on the marginal distribution, so it is not "
                 "comparable across studies without their class prevalences. "
                 "Balanced accuracy averages per-class recall and is reported because "
                 "the class distributions differ.")
    lines.append("")
    if len(metrics) >= 2:
        first, second = metrics.iloc[0], metrics.iloc[1]
        lines.append(
            f"The studies' 95% intervals overlap (accuracy "
            f"{_fmt(first['accuracy_ci_low'])}-{_fmt(first['accuracy_ci_high'])} vs "
            f"{_fmt(second['accuracy_ci_low'])}-{_fmt(second['accuracy_ci_high'])}; "
            f"kappa {_fmt(first['kappa_ci_low'])}-{_fmt(first['kappa_ci_high'])} vs "
            f"{_fmt(second['kappa_ci_low'])}-{_fmt(second['kappa_ci_high'])}), and no "
            "between-study hypothesis test was performed: the samples are "
            "independent and the tasks differ, so the accuracy and kappa differences "
            "are descriptive, not estimated effects.")
        reference_descriptions = {
            cfg["run"]["run_id"]:
            (cfg.get("report", {}).get("reference_description") or "not described"
             ).rstrip(".")
            for cfg in configs}
        lines.append("")
        lines.append(
            "The reference is constructed differently in each study ("
            + "; ".join(f"`{name}`: {reference_descriptions.get(name, 'not described')}"
                        for name in metrics["dataset"])
            + "), so the two accuracy figures are not directly comparable.")
    lines.append("")

    lines.append("## 3. Per-class behaviour")
    lines.append("")
    lines.append("| Dataset | Class | n | Precision | Recall (95% CI) | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in classes.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['class']} | {int(row['n_reference'])} | "
            f"{_fmt(row['precision'])} | {_fmt(row['recall'])} "
            f"({_fmt(row['recall_ci_low'])}-{_fmt(row['recall_ci_high'])}) | "
            f"{_fmt(row['f1'])} |")
    lines.append("")

    lines.append("## 4. Disagreement structure")
    lines.append("")
    lines.append("| Dataset | Agree | Adjacent | Opposite | Disagreements |")
    lines.append("|---|---|---|---|---|")
    for dataset in metrics["dataset"]:
        subset = severity[severity["dataset"] == dataset].set_index("bucket")
        values = []
        for bucket in ("agree", "adjacent", "opposite"):
            share = (float(subset.loc[bucket, "share"]) if bucket in subset.index else 0.0)
            values.append(_pct(share))
        count = int(metrics.loc[metrics["dataset"] == dataset, "n_disagreements"].iloc[0])
        lines.append(f"| {dataset} | " + " | ".join(values) + f" | {count} |")
    lines.append("")
    lines.append("Adjacency is only meaningful for ordered categories. For a binary "
                 "task every disagreement is at maximum distance; for unordered "
                 "categories with three or more classes the distance buckets follow "
                 "the declaration order and should not be read as adjacent or "
                 "opposite. The `Agree`, `Adjacent` and `Opposite` columns are shares "
                 "of each study's analysed rows (they sum to 100%), not of its "
                 "disagreements; the `Disagreements` column is a count.")
    lines.append("")

    lines.append("### Where disagreements go")
    lines.append("")
    lines.append("| Dataset | Reference -> prediction | n | Share of disagreements |")
    lines.append("|---|---|---|---|")
    for dataset in metrics["dataset"]:
        subset = pairs[pairs["dataset"] == dataset].sort_values("n", ascending=False)
        for _, row in subset.head(6).iterrows():
            lines.append(f"| {dataset} | {row['reference']} -> {row['predicted']} | "
                         f"{int(row['n'])} | {_pct(row['share_of_disagreements'])} |")
    lines.append("")
    lines.append("The table lists the most frequent disagreement directions per "
                 "dataset; the full table is `cross_dataset_pairs.csv`.")
    lines.append("")

    lines.append("### Observation length (exploratory)")
    lines.append("")
    lines.append("| Dataset | Median words, agree | Median words, disagree | "
                 "Permutation p |")
    lines.append("|---|---|---|---|")
    for _, row in text.iterrows():
        lines.append(f"| {row['dataset']} | {_fmt(row['median_words_agree'], 1)} | "
                     f"{_fmt(row['median_words_disagree'], 1)} | "
                     f"{_fmt(row['length_p'])} |")
    lines.append("")
    lines.append("One pre-specified feature, seeded permutation test, no multiplicity "
                 "adjustment. A non-significant result is not evidence of no effect; "
                 "the samples are small.")
    lines.append("")

    if len(references):
        lines.append("### Secondary reference structure")
        lines.append("")
        lines.append("| Dataset | Comparison | n | Agreement | Kappa |")
        lines.append("|---|---|---|---|---|")
        for _, row in references.iterrows():
            lines.append(f"| {row['dataset']} | {row['comparison']} | {int(row['n'])} | "
                         f"{_fmt(row['agreement'])} | {_fmt(row['kappa_unweighted'])} |")
        lines.append("")
        lines.append("These rows describe how the references relate to each "
                     "other and to the model. They show that some disagreements sit "
                     "between references; they do not confirm a model error. A "
                     "secondary reference may be a proxy label rather than a second "
                     "human annotation.")
        lines.append("")

    if comparative:
        lines.extend(_qualitative_section(qualitative, metrics))
    interpretation_number = 6 if comparative else 5

    lines.append(f"## {interpretation_number}. Interpretation")
    lines.append("")
    for _, row in metrics.iterrows():
        lines.append(
            f"- **{row['dataset']}**: accuracy {_fmt(row['accuracy'])} (balanced "
            f"{_fmt(row['balanced_accuracy'])}) with kappa "
            f"{_fmt(row['kappa_unweighted'])}; majority class "
            f"`{row['majority_class']}` accounts for {_pct(row['majority_share'])} of "
            "the analysed rows.")
    if len(metrics):
        weakest = metrics.sort_values("balanced_accuracy").iloc[0]
        lines.append(f"- The lowest balanced accuracy was on `{weakest['dataset']}` "
                     f"({_fmt(weakest['balanced_accuracy'])}). Balanced accuracy "
                     "averages per-class recall, so a lower value means at least one "
                     "category was recalled less well; it does not by itself identify "
                     "which category, how many rows, or why.")
    by_id = {entry["run_id"]: entry for entry in qualitative if entry}
    concentrated = [
        f"`{name}` {reason_label_lower(by_id[name]['contrast'].iloc[0]['reason'])}"
        for name in metrics["dataset"] if name in by_id
        and len(by_id[name]["contrast"])]
    if comparative and concentrated:
        lines.append("- Qualitatively, the reason most concentrated in disagreements "
                     "was " + "; ".join(concentrated) + ". These are hypotheses about "
                     "one sample each, not established causes.")
    notes = []
    for cfg in configs:
        results_dir = Path(cfg["_meta"]["output_dir"]) / "results"
        run_id = cfg["run"]["run_id"]
        verdict = hypothesis_summary(cfg, results_dir)
        if verdict:
            notes.append(f"- **`{run_id}`** — {verdict}")
        three_way = three_way_summary(cfg, results_dir)
        if three_way:
            notes.append(f"- **`{run_id}`** — {three_way}")
    lines.extend(notes)
    lines.append("- Higher accuracy on one study than another is not evidence that the "
                 "task, the model, or the benchmark is easier. Task, class count, class "
                 "prevalence, reference construction, and sample selection differ "
                 "together.")
    lines.append("- Kappa differences partly reflect prevalence differences: a dominant "
                 "majority class changes chance agreement, so kappa is not a "
                 "distribution-free effect size.")
    lines.append("- No causal claim is made anywhere in this report. All statements are "
                 "descriptive associations within one sample per study.")
    lines.append("")

    lines.append(f"## {interpretation_number + 1}. Limitations")
    lines.append("")
    limitations = [
        "- The reference is a benchmark label, not an adjudicated gold "
        "standard. Unless per-annotator labels are released, human-human kappa "
        "cannot be computed and model errors cannot be confirmed.",
        "- Accuracy and recall use Wilson intervals; kappa, balanced accuracy, "
        "precision and F1 use a seeded percentile bootstrap.",
        "- All intervals are conditional on this sample; the sample is not "
        "stratified.",
        "- The studies are independent samples with different tasks, class counts, "
        "reference construction, class prevalence and sample size; their intervals "
        "overlap and no between-study test was performed, so between-study "
        "differences are descriptive.",
        "- No causal claim is made. Statements are descriptive associations "
        "within one sample per study.",
        "- Temperature is not controlled and model versions drift.",
    ]
    for cfg in configs:
        for limitation in cfg.get("report", {}).get("limitations", []):
            limitations.append(f"- `{cfg['run']['run_id']}`: {limitation}")
    for _, row in metrics.iterrows():
        limitations.append(f"- `{row['dataset']}`: {row['n']} analysed rows; all "
                           "intervals are conditional on this sample.")
    limitations.append("- Temperature was not controlled; stability was checked only "
                       "on a five-row pilot in one study.")
    lines.extend(dedupe_limitations(limitations, _LIMITATION_PREFIX))
    lines.append("")

    lines.append(f"## {interpretation_number + 2}. Reproducibility")
    lines.append("")
    portable = {cfg["run"]["run_id"]: _portable_path(cfg["_meta"]["config_path"])
                for cfg in configs}
    lines.append("```")
    for cfg in configs:
        config_arg = portable[cfg["run"]["run_id"]] or "<config>"
        lines.append(f"uv run python scripts/05_evaluate.py --config {config_arg}")
        lines.append(f"uv run python scripts/06_disagreements.py --config {config_arg}")
        lines.append(f"uv run python scripts/08_qualitative.py --config {config_arg} "
                     f"--settings runs/{cfg['run']['run_id']}/qualitative/settings.yaml "
                     "--report")
    lines.append("uv run python scripts/07_compare.py --configs \\")
    config_args = [portable[cfg["run"]["run_id"]] or "<config>" for cfg in configs]
    for position, config_arg in enumerate(config_args):
        continuation = " \\" if position < len(config_args) - 1 else ""
        lines.append(f"    {config_arg}{continuation}")
    lines.append("```")
    lines.append("")
    lines.append("Regenerate the per-study PDFs with "
                 "`uv run python scripts/09_report.py --run-id <run_id>`.")
    lines.append("")
    lines.append("Seeds, input hashes, and the exact model string are recorded in each "
                 "run's `manifest.json` and `results/run_metadata.json`.")
    lines.append("")
    lines.append("Methodological decisions and implementation mistakes are recorded in "
                 "`MISTAKES.md` at the project root and in `SPEC.md`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(config_paths: list[str | Path], output_dir: str | Path = "comparison") -> int:
    configs = [load_config(path) for path in config_paths]
    out = Path(output_dir)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[3] / out
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(message: str = "") -> None:
        print(message)
        log_lines.append(message)

    log(f"script: general-agent compare (version {__version__})")
    log(f"started: {now_iso()}")
    log(f"configs: {len(configs)}")

    rows, classes_frames, severity_frames, reference_frames = [], [], [], []
    pair_frames, text_rows = [], []
    qualitative: list[dict | None] = []
    for cfg in configs:
        row, classes, severity, references, pairs, text = collect_study(cfg)
        rows.append(row)
        classes_frames.append(classes)
        severity_frames.append(severity)
        reference_frames.append(references)
        pair_frames.append(pairs)
        text_rows.append(text)
        summary = _qualitative_summary(cfg)
        qualitative.append(summary)
        log(f"collected: {row['dataset']} (n={row['n']}, classes={row['n_classes']}, "
            f"qualitative={'yes' if summary else 'no'})")

    metrics = pd.DataFrame(rows)
    classes = pd.concat(classes_frames, ignore_index=True)
    severity = pd.concat(severity_frames, ignore_index=True)
    references = pd.concat(reference_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    text = pd.DataFrame(text_rows)

    metrics.to_csv(tables / "cross_dataset_metrics.csv", index=False, float_format="%.6f")
    classes.to_csv(tables / "cross_dataset_classes.csv", index=False, float_format="%.6f")
    severity.to_csv(tables / "cross_dataset_severity.csv", index=False, float_format="%.6f")
    references.to_csv(tables / "cross_dataset_references.csv", index=False, float_format="%.6f")
    pairs.to_csv(tables / "cross_dataset_pairs.csv", index=False, float_format="%.6f")
    text.to_csv(tables / "cross_dataset_text.csv", index=False, float_format="%.6f")

    _figure_metrics(metrics, figures / "fig1_accuracy_kappa.svg")
    _figure_classes(classes, figures / "fig2_class_recall.svg")
    _figure_severity(severity, figures / "fig3_severity.svg")

    _report(metrics, classes, severity, references, pairs, text, configs,
            out / "report.md", qualitative)

    metadata = {
        "finished_at": now_iso(),
        "tool": {"name": "general-agent", "version": __version__},
        "configs": [{"path": cfg["_meta"]["config_path"],
                     "sha256": cfg["_meta"]["config_sha256"],
                     "run_id": cfg["run"]["run_id"],
                     "results_tables": {
                         name: file_sha256(Path(cfg["_meta"]["output_dir"]) / "results" / name)
                         for name in REQUIRED
                         if (Path(cfg["_meta"]["output_dir"]) / "results" / name).exists()
                     }} for cfg in configs],
        "tables": ["cross_dataset_metrics.csv", "cross_dataset_classes.csv",
                   "cross_dataset_severity.csv", "cross_dataset_references.csv",
                   "cross_dataset_pairs.csv", "cross_dataset_text.csv"],
        "figures": ["fig1_accuracy_kappa.svg", "fig2_class_recall.svg",
                    "fig3_severity.svg"],
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "comparison_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(f"tables: {tables}")
    log(f"figures: {figures}")
    log(f"report: {out / 'report.md'}")
    print(f"finished: {now_iso()}")
    return 0
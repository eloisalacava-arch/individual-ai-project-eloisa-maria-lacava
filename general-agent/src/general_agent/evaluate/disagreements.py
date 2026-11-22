"""Stage 06 - disagreement analysis (post-evaluation, descriptive).

Reads the frozen predictions, the sealed reference (already opened by stage 05),
and the blind text. Writes:

  t11_disagreement_pairs.csv      reference -> predicted counts for every pair
  t12_text_profile.csv            observation length/readability by agreement group
  t12b_length_test.csv            seeded permutation test on observation length
  t13_reference_agreement.csv     agreement with any declared extra reference
  t14_subgroup_tests.csv          pre-specified subgroup hypothesis contrasts
  t15_three_way_agreement.csv     AI vs primary vs extra-reference decomposition
  t16_reference_dispute.csv       human-reference disagreement rate by subgroup
  disagreement_log.txt

Nothing here is causal. Length differences are reported as associations and the
test is labelled exploratory. Subgroup contrasts are observational: the flags are
review characteristics, not randomised treatments.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import class_names, load_config
from ..discover import map_labels
from ..provenance import file_sha256, now_iso, update_manifest
from . import metrics, subgroups

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SYLLABLE = re.compile(r"[aeiouy]+")


def _text_metrics(text: str) -> dict:
    text = str(text).replace("<br />", " ").replace("<br/>", " ")
    words = _WORD.findall(text)
    sentences = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    words_per_sentence = [len(_WORD.findall(part)) for part in sentences]
    words_per_sentence = [value for value in words_per_sentence if value > 0]
    syllables = 0
    for word in words:
        groups = _SYLLABLE.findall(word.lower())
        count = len(groups)
        if word.lower().endswith("e") and count > 1 and not word.lower().endswith(("le", "ee")):
            count -= 1
        syllables += max(count, 1)
    n_words = len(words)
    n_sentences = len(words_per_sentence)
    flesch = (
        206.835
        - 1.015 * (n_words / n_sentences if n_sentences else math.nan)
        - 84.6 * (syllables / n_words if n_words else math.nan)
    )
    return {
        "chars": len(text),
        "words": n_words,
        "sentences": n_sentences,
        "mean_sentence_words": float(np.mean(words_per_sentence)) if words_per_sentence else math.nan,
        "flesch": flesch,
    }


def _permutation_p(a: np.ndarray, b: np.ndarray, draws: int, seed: int) -> float:
    """Two-sided permutation test on the difference in means."""
    if len(a) == 0 or len(b) == 0:
        return math.nan
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(draws):
        rng.shuffle(pooled)
        statistic = abs(float(pooled[: len(a)].mean() - pooled[len(a):].mean()))
        if statistic >= observed:
            count += 1
    return (count + 1) / (draws + 1)


def _kappa_pair(left: pd.Series, right: pd.Series, classes: list[str],
                ordered: bool) -> dict:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) == 0:
        return {"n": 0, "agreement": math.nan, "kappa_unweighted": math.nan,
                "kappa_quadratic": math.nan}
    y_left = metrics.label_indices(frame["left"], classes)
    y_right = metrics.label_indices(frame["right"], classes)
    matrix = metrics.confusion_matrix(y_left, y_right, len(classes))
    return {
        "n": int(len(frame)),
        "agreement": float(np.trace(matrix) / len(frame)),
        "kappa_unweighted": metrics.cohen_kappa(matrix),
        "kappa_quadratic": metrics.weighted_kappa(matrix, "quadratic") if ordered else math.nan,
    }


def run_disagreements(config_path: str | Path) -> int:
    cfg = load_config(config_path)
    out_dir = Path(cfg["_meta"]["output_dir"])
    results = out_dir / "results"
    classes = class_names(cfg)
    if not results.exists():
        print(f"ABORT: {results} is missing; run stage 05 (evaluate) first")
        return 1

    lines: list[str] = []

    def log(message: str = "") -> None:
        print(message)
        lines.append(message)

    log("script: general-agent disagreements (stage 06)")
    log(f"started: {now_iso()}")
    log(f"run_id: {cfg['run']['run_id']} | classes: {classes}")

    frozen = pd.read_csv(results / "frozen_predictions.csv", dtype="string",
                         keep_default_na=False)
    # Rename the prediction column first: the reference may also be called
    # "label", and pandas would suffix both to label_x / label_y.
    frozen = frozen.rename(columns={"label": "prediction"})
    benchmark = pd.read_csv(out_dir / "sealed" / "benchmark.csv", dtype="string",
                            keep_default_na=False)
    blind_path = out_dir / "blind" / "blind.csv"
    if blind_path.exists():
        blind = pd.read_csv(blind_path, dtype="string", keep_default_na=False)
    else:
        blind = pd.DataFrame(columns=["id", "text"])
        log("WARN: blind text is missing; text-profile tables will be empty")
    log(f"rows: frozen {len(frozen)}, sealed reference {len(benchmark)}, "
        f"blind text {len(blind)}")

    frame = frozen.merge(benchmark, on="id", how="inner", validate="one_to_one")
    if "text" in blind.columns and len(blind):
        frame = frame.merge(blind, on="id", how="inner", validate="one_to_one")
    else:
        frame["text"] = ""
    required = ["id", "status", "prediction", "text"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"disagreement join is missing columns {missing}; got {list(frame.columns)}"
        )
    log(f"rows: {len(frozen)} -> {len(frame)} (reason: one-to-one join on id)")

    reference_columns = [cfg["benchmark"]["label_column"]] + [
        c for c in cfg["benchmark"].get("extra_label_columns", [])
    ]
    label_map = cfg["benchmark"].get("label_map")
    for column in reference_columns:
        mapped, unmapped = map_labels(frame[column], label_map, classes)
        if unmapped:
            log(f"WARN: {column} has unmapped values {unmapped}; those rows are excluded "
                "from its comparison")
        frame[column] = mapped

    primary = cfg["benchmark"]["label_column"]
    frame = frame[frame["status"] == "ok"].copy()
    frame = frame[frame[primary].notna()].copy()
    log(f"rows: {len(frame)} analysed (reason: status ok and a usable primary label)")

    y_true = metrics.label_indices(frame[primary], classes)
    y_pred = metrics.label_indices(frame["prediction"], classes)
    k = len(classes)
    agree = y_true == y_pred

    # ---- t11: reference -> predicted pairs ------------------------------
    pairs = frame.assign(
        ordinal_distance=np.abs(y_true - y_pred),
        agreed=agree,
    ).groupby([primary, "prediction"], dropna=False).size().reset_index(name="n")
    pairs = pairs.rename(columns={primary: "reference", "prediction": "predicted"})
    pairs["ordinal_distance"] = np.abs(
        [classes.index(r) - classes.index(p) for r, p in
         zip(pairs["reference"], pairs["predicted"])])
    n_disagreements = int((~agree).sum())
    pairs["share_of_disagreements"] = np.where(
        pairs["ordinal_distance"] > 0,
        pairs["n"] / n_disagreements if n_disagreements else 0.0,
        0.0)
    pairs["share_of_all"] = pairs["n"] / len(frame)
    pairs = pairs.sort_values(["ordinal_distance", "n"], ascending=[False, False])
    pairs.to_csv(results / "t11_disagreement_pairs.csv", index=False,
                 float_format="%.6f")

    # ---- t12: text profile ----------------------------------------------
    text_features = pd.DataFrame([_text_metrics(value) for value in frame["text"]])
    profile = pd.concat([frame[["id"]].reset_index(drop=True),
                         text_features.reset_index(drop=True),
                         pd.Series(agree, name="agreed")], axis=1)
    profile["group"] = np.where(profile["agreed"], "agree", "disagree")
    summary = profile.groupby("group").agg(
        n=("chars", "size"),
        median_chars=("chars", "median"),
        mean_chars=("chars", "mean"),
        median_words=("words", "median"),
        mean_words=("words", "mean"),
        median_sentence_words=("mean_sentence_words", "median"),
        median_flesch=("flesch", "median"),
    ).reset_index()
    summary.to_csv(results / "t12_text_profile.csv", index=False, float_format="%.6f")

    agree_words = profile.loc[profile["group"] == "agree", "words"].to_numpy(float)
    disagree_words = profile.loc[profile["group"] == "disagree", "words"].to_numpy(float)
    p_value = _permutation_p(agree_words, disagree_words,
                             draws=10_000, seed=int(cfg["run"]["seed"]))
    test = pd.DataFrame([{
        "feature": "observation_words",
        "n_agree": len(agree_words),
        "n_disagree": len(disagree_words),
        "mean_agree": float(agree_words.mean()) if len(agree_words) else math.nan,
        "mean_disagree": float(disagree_words.mean()) if len(disagree_words) else math.nan,
        "difference_disagree_minus_agree":
            float(disagree_words.mean() - agree_words.mean())
            if len(agree_words) and len(disagree_words) else math.nan,
        "permutation_draws": 10_000,
        "p_two_sided": p_value,
        "seed": int(cfg["run"]["seed"]),
        "status": "exploratory; one pre-specified feature; no multiplicity adjustment",
    }])
    test.to_csv(results / "t12b_length_test.csv", index=False, float_format="%.6f")

    # ---- t13: agreement with extra references ---------------------------
    extra_rows = []
    for column in reference_columns[1:]:
        if column not in frame.columns:
            continue
        values = frame[column]
        available = values.notna()
        for name, left, right in (
            ("ai_vs_" + column, frame["prediction"], values),
            (primary + "_vs_" + column, frame[primary], values),
        ):
            stats = _kappa_pair(left[available], right[available], classes,
                                bool(cfg["labels"].get("ordered")))
            extra_rows.append({
                "comparison": name,
                **stats,
                "note": "secondary reference; disagreements are reference-to-reference "
                        "structure, not confirmed model errors",
            })
    extra_table = pd.DataFrame(extra_rows, columns=[
        "comparison", "n", "agreement", "kappa_unweighted", "kappa_quadratic", "note"])
    extra_table.to_csv(results / "t13_reference_agreement.csv", index=False,
                       float_format="%.6f")

    # ---- t14/t15/t16: subgroups and three-way reference structure --------
    level = float(cfg["evaluation"]["ci_level"])
    subgroup_config = cfg["evaluation"]
    subgroup_columns = [column for column in
                        (subgroup_config.get("subgroup_columns") or [])
                        if column in frame.columns]
    subgroup_reference = subgroup_config.get("subgroup_reference")
    composites = subgroup_config.get("subgroup_composites") or {}
    extra_references = [c for c in reference_columns[1:] if c in frame.columns]

    tests = subgroups.subgroup_tests(
        frame, reference_columns, "prediction", subgroup_columns,
        subgroup_reference, composites, level)
    three_way = subgroups.three_way_agreement(
        frame, primary, extra_references, "prediction", level)
    disputes = subgroups.reference_disputes(
        frame, primary, extra_references, subgroup_columns,
        subgroup_reference, composites, level)

    written = []
    if not tests.empty:
        tests.to_csv(results / "t14_subgroup_tests.csv", index=False,
                     float_format="%.6f")
        written.append("t14_subgroup_tests.csv")
    if not three_way.empty:
        three_way.to_csv(results / "t15_three_way_agreement.csv", index=False,
                         float_format="%.6f")
        written.append("t15_three_way_agreement.csv")
    if not disputes.empty:
        disputes.to_csv(results / "t16_reference_dispute.csv", index=False,
                        float_format="%.6f")
        written.append("t16_reference_dispute.csv")

    # ---- log ------------------------------------------------------------
    log("")
    log(f"disagreements: {n_disagreements} of {len(frame)}")
    top = pairs[pairs["ordinal_distance"] > 0].head(5)
    for _, row in top.iterrows():
        log(f"  {row['reference']} -> {row['predicted']}: {int(row['n'])} "
            f"({row['share_of_disagreements']:.1%} of disagreements)")
    medians = summary.set_index("group")["median_words"]
    log(f"text length (words): agree median {medians.get('agree', math.nan)}, "
        f"disagree median {medians.get('disagree', math.nan)}; "
        f"permutation p = {p_value:.4f} (exploratory)")
    for row in extra_rows:
        log(f"  {row['comparison']}: n={row['n']} agreement={row['agreement']:.4f} "
            f"kappa={row['kappa_unweighted']:.4f}")
    log("- Extra-reference comparisons are reference-to-reference structure; they do "
        "not turn a disagreement into a confirmed model error.")
    if not tests.empty:
        log("")
        log("== Subgroup hypothesis contrasts (observational) ==")
        for _, row in tests.iterrows():
            log(f"  {row['reference']} | {row['subgroup']} ({row['family']}): "
                f"rate {row['rate_subgroup']:.4f} vs {row['rate_reference']:.4f}; "
                f"RD {row['risk_difference']:+.4f} "
                f"({row['rd_ci_low']:+.4f}, {row['rd_ci_high']:+.4f}); "
                f"fisher p={row['fisher_p']:.4f}; q={row['bh_q']}; {row['verdict']}")
    if not three_way.empty:
        log("")
        log("== Three-way agreement (AI vs primary vs extra reference) ==")
        for _, row in three_way.iterrows():
            log(f"  {row['comparison']} | {row['category']}: {int(row['n'])} "
                f"({row['share']:.1%})")
    if not disputes.empty:
        log("")
        log("== Human-reference disagreement by subgroup ==")
        for _, row in disputes.iterrows():
            log(f"  {row['comparison']} | {row['subgroup']}: dispute rate "
                f"{row['rate_subgroup']:.4f} vs {row['rate_reference']:.4f}; "
                f"RD {row['risk_difference']:+.4f} "
                f"({row['rd_ci_low']:+.4f}, {row['rd_ci_high']:+.4f}); "
                f"fisher p={row['fisher_p']:.4f}")
    log("- Binary data has no 'adjacent' step: every error is at maximum ordinal "
        "distance, so severity labels are meaningful only for 3+ ordered classes.")

    (results / "disagreement_log.txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    update_manifest(out_dir, "06_disagreements", {
        "finished_at": now_iso(),
        "analysed_rows": int(len(frame)),
        "n_disagreements": n_disagreements,
        "tables": ["t11_disagreement_pairs.csv", "t12_text_profile.csv",
                   "t12b_length_test.csv", "t13_reference_agreement.csv",
                   "disagreement_log.txt"] + written,
        "length_test_p": p_value,
        "extra_references": reference_columns[1:],
        "subgroup_columns": subgroup_columns,
        "subgroup_reference": subgroup_reference,
        "blind_sha256": file_sha256(out_dir / "blind" / "blind.csv"),
    })
    log(f"written to {results}")
    print(f"finished: {now_iso()}")
    return 0
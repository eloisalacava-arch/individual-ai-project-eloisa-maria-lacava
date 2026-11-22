"""Stage 06: pair table, text profile, secondary reference structure."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from general_agent.evaluate.analysis import run_evaluation
from general_agent.evaluate.disagreements import run_disagreements
from test_evaluate import _setup


def test_disagreement_tables_are_written_and_reconcile(tmp_path):
    errors = {0: "neutral", 1: "negative", 2: "neutral",
              3: "neutral", 4: "positive", 5: "neutral"}
    config_path, run_dir = _setup(tmp_path, errors=errors)
    assert run_evaluation(config_path) == 0
    assert run_disagreements(config_path) == 0

    results = run_dir / "results"
    for name in ("t11_disagreement_pairs.csv", "t12_text_profile.csv",
                 "t12b_length_test.csv", "t13_reference_agreement.csv",
                 "disagreement_log.txt"):
        assert (results / name).exists(), name

    pairs = pd.read_csv(results / "t11_disagreement_pairs.csv")
    assert int(pairs.loc[pairs["ordinal_distance"] > 0, "n"].sum()) == 6
    assert abs(float(pairs["share_of_all"].sum()) - 1.0) < 1e-4  # CSV rounding

    profile = pd.read_csv(results / "t12_text_profile.csv")
    assert set(profile["group"]) == {"agree", "disagree"}
    assert int(profile["n"].sum()) == 30

    test = pd.read_csv(results / "t12b_length_test.csv").iloc[0]
    assert 0.0 < float(test["p_two_sided"]) <= 1.0
    assert test["permutation_draws"] == 10000

    references = pd.read_csv(results / "t13_reference_agreement.csv")
    assert len(references) == 0


def test_secondary_reference_comparisons(tmp_path):
    config_path, run_dir = _setup(tmp_path, errors={0: "positive"})

    # Add a secondary reference to the sealed file and the config.
    sealed_path = run_dir / "sealed" / "benchmark.csv"
    sealed = pd.read_csv(sealed_path)
    sealed["aux"] = sealed["sentiment"]
    sealed.loc[0, "aux"] = "positive" if sealed.loc[0, "sentiment"] != "positive" else "negative"
    sealed.to_csv(sealed_path, index=False)

    cfg = yaml.safe_load(config_path.read_text())
    cfg["benchmark"]["extra_label_columns"] = ["aux"]
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    assert run_evaluation(config_path) == 0
    assert run_disagreements(config_path) == 0

    references = pd.read_csv(run_dir / "results" / "t13_reference_agreement.csv")
    assert set(references["comparison"]) == {"ai_vs_aux", "sentiment_vs_aux"}
    assert int(references["n"].min()) == 30
    for value in references["kappa_unweighted"]:
        assert -1.0 <= float(value) <= 1.0

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "06_disagreements" in manifest["stages"]

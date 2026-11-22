"""Report polish: reference punctuation, limitation de-duplication, and the
zero-disagreement qualitative message.

No model call: every provider is a deterministic stub.
"""

from __future__ import annotations

import pandas as pd

from general_agent.audit import run_audit
from general_agent.blind import run_blind
from general_agent.classify import approve_pilot
from general_agent.classify import run as classify_run
from general_agent.config import load_config
from general_agent.evaluate.analysis import run_evaluation
from general_agent.evaluate.disagreements import run_disagreements
from general_agent.limitations import dedupe_limitations
from general_agent.report import compose
from qualitative_helpers import evaluated_run
from stubs import StubProvider
from ui_helpers import make_ui_study


def _zero_disagreement_run(tmp_path):
    """A studied run whose every prediction equals the reference label."""
    import re

    study = make_ui_study(tmp_path, run_id="zero_disagreements", classes=2, n=6)
    labels = study["classes"]

    def answer_correctly(provider, prompt):
        match = re.search(r"observation (\d+)", prompt)
        if match:
            provider.label = labels[int(match.group(1)) % len(labels)]

    config_path = study["config_path"]
    assert run_audit(config_path) == 0
    assert run_blind(config_path) == 0
    stub = StubProvider(load_config(config_path), label=labels[0],
                        on_call=answer_correctly)
    assert classify_run(config_path, mode="pilot", provider=stub) == 0
    assert approve_pilot(config_path) == 0
    assert classify_run(config_path, mode="full", provider=stub) == 0
    assert run_evaluation(config_path) == 0
    assert run_disagreements(config_path) == 0
    return study


# --------------------------------------------------------------------------- #
# 1. Reference punctuation
# --------------------------------------------------------------------------- #

def test_reference_description_never_doubles_the_period(tmp_path):
    study = evaluated_run(tmp_path, classes=2, n=8,
                          reference_description="Human labels from `x.csv`.")
    markdown, _ = compose.compose_study(study["config_path"])
    assert ".." not in markdown
    assert "Reference: Human labels from `x.csv`." in markdown
    # The metadata table keeps exactly one terminal mark too.
    assert "| Reference | Human labels from `x.csv`. |" in markdown


# --------------------------------------------------------------------------- #
# 2. Limitations de-duplication
# --------------------------------------------------------------------------- #

def test_dedupe_keeps_each_caveat_once_without_losing_details():
    lines = [
        "The reference is a benchmark label, not an adjudicated gold standard; a disagreement is not a confirmed model error.",
        "Accuracy and recall use Wilson intervals; kappa, balanced accuracy, precision and F1 use a seeded percentile bootstrap.",
        "All intervals are conditional on this sample; the sample is not stratified.",
        "No causal claim is made. Statements are descriptive associations within one sample.",
        "Temperature is not controlled and model versions drift.",
        "Human labels are a benchmark, not adjudicated truth; a disagreement is not a confirmed model error.",
        "The sample is not stratified; class prevalence reflects the uploaded file.",
        "Model temperature is not controlled; repeated runs may differ.",
        "Rows are assumed independent; duplicate observations may inflate apparent precision.",
        "22 analysed rows; every interval is conditional on this sample.",
    ]
    joined = "\n".join(dedupe_limitations(lines))
    assert "Human labels are a benchmark" not in joined
    assert "Model temperature is not controlled" not in joined
    assert "22 analysed rows" not in joined
    # Unique details survive.
    assert "prevalence reflects the uploaded file" in joined
    assert "Rows are assumed independent" in joined
    assert "not stratified" in joined
    assert joined.count("not stratified") == 1
    assert joined.count("Temperature is not controlled") == 1


def test_study_report_limitations_have_no_restatements(tmp_path):
    study = evaluated_run(tmp_path, classes=3, n=15)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "Human labels are a benchmark, not adjudicated truth" not in markdown
    assert "Model temperature is not controlled" not in markdown
    assert "Rows are assumed independent" in markdown
    assert "Class prevalence reflects the uploaded file" in markdown
    assert markdown.count("not stratified") == 1
    assert markdown.count("Temperature is not controlled") == 1
    assert "15 analysed rows; every interval" not in markdown
    assert ".." not in markdown


def test_comparison_report_limitations_are_deduplicated(tmp_path):
    study_a = evaluated_run(tmp_path, classes=2, n=8, run_id="study_a")
    study_b = evaluated_run(tmp_path, classes=2, n=8, run_id="study_b")
    out = tmp_path / "comparison"
    from general_agent.evaluate.compare import run_comparison
    assert run_comparison([study_a["config_path"], study_b["config_path"]],
                          output_dir=out) == 0
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "Human labels are a benchmark, not adjudicated truth" not in markdown
    assert "Model temperature is not controlled" not in markdown
    assert "not stratified" in markdown
    assert markdown.count("not stratified") == 1
    assert "Temperature is not controlled" in markdown
    assert "Class prevalence reflects the uploaded file" in markdown


# --------------------------------------------------------------------------- #
# 3. Zero disagreements
# --------------------------------------------------------------------------- #

def test_zero_disagreements_states_qualitative_was_skipped(tmp_path):
    study = _zero_disagreement_run(tmp_path)
    markdown, _ = compose.compose_study(study["config_path"])
    assert markdown.count("disagreements: **0**") == 1
    assert "skipped" in markdown.lower()
    assert "there were no disagreements" in markdown
    assert "skipped (no disagreements)" in markdown
    assert "Qualitative analysis has not been run" not in markdown


def test_nonzero_disagreements_keep_the_run_stage_08_message(tmp_path):
    # The deterministic stub always answers the first class, so a three-class
    # benchmark yields disagreements and the "run stage 08" wording remains.
    study = evaluated_run(tmp_path, classes=3, n=12)
    markdown, _ = compose.compose_study(study["config_path"])
    assert "Qualitative analysis has not been run" in markdown
    assert "there were no disagreements" not in markdown

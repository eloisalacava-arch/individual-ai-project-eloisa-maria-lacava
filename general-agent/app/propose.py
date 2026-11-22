"""Automatic proposals for the simple three-step flow.

Reads the two uploaded files only. Column discovery reuses the engine's
``discover.propose`` so the interface asks exactly the questions the audit would
ask. Category names and mappings are *proposed* for the user to confirm; label
polarity is never inferred, and numeric or messy label values are flagged for an
explicit mapping.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from general_agent import discover

from app import registry
from app.config_builder import PILOT_N, suggest_max_calls
from app.definitions import PLACEHOLDER_TEXT

SLUG = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
EMPTY_DATASET_CFG = {
    "dataset": {"id_column": None, "text_column": None, "text_template": None},
    "benchmark": {
        "id_column": None, "label_column": None, "label_map": None,
        "extra_label_columns": [], "on_unknown_label": "error",
        "on_id_mismatch": "error", "allow_duplicate_ids": False,
    },
    "labels": {"classes": [
        {"name": "class-a", "definition": "placeholder"},
        {"name": "class-b", "definition": "placeholder"},
    ]},
    "sample": {},
}


@dataclass
class Problem:
    field: str
    severity: str          # blocking | decision | warning
    message: str
    fix_hint: str = ""
    options: list[dict] = field(default_factory=list)


@dataclass
class Proposal:
    dataset_id_column: str | None = None
    text_column: str | None = None
    benchmark_id_column: str | None = None
    label_column: str | None = None
    classes: list[dict] = field(default_factory=list)
    label_map: dict = field(default_factory=dict)
    ordered: bool = False
    provider_id: str | None = None
    model_name: str | None = None
    rows_mode: str = "all"
    sample_n: int | None = None
    seed: int = 20261116
    max_calls: int = 40
    max_usd: float = 0.5
    qualitative_enabled: bool = True
    qualitative_sample_n: int = 15
    qualitative_control_n: int = 5
    qualitative_max_calls: int = 30
    qualitative_max_usd: float = 0.25
    task_role: str = ""
    decision_rule: str = ""
    report_title: str = ""
    problems: list[Problem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def blocking(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "blocking"]

    def decisions(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "decision"]

    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]


def suggest_max_usd(rows: int, price: dict, *, per_prompt_chars: int = 1500,
                    output_tokens: int = 300, headroom: float = 5.0,
                    minimum: float = 0.10, maximum: float = 10.0) -> float:
    """A bounded spend cap with headroom over the character-based floor."""
    per_call = ((per_prompt_chars / 4.0) * float(price.get("input", 0.0))
                + output_tokens * float(price.get("output", 0.0))) / 1_000_000
    projected = max(int(rows), 0) * per_call * headroom
    return round(min(max(max(projected, minimum), minimum), maximum), 2)


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip().lower())
    return text.strip("-._")


def _propose_classes(values: list[str]) -> tuple[list[dict], dict, bool]:
    """Return (classes, value->name map, used_generic_names)."""
    classes: list[dict] = []
    mapping: dict[str, str] = {}
    used: set[str] = set()
    generic = False
    for index, value in enumerate(values):
        candidate = _slug(value)
        if SLUG.match(candidate) and len(candidate) >= 2 and not candidate.isdigit() \
                and candidate not in used:
            name = candidate
        else:
            name = f"category-{index + 1}"
            generic = True
        base, counter = name, 2
        while name in used:
            name = f"{base}-{counter}"
            counter += 1
        used.add(name)
        classes.append({
            "name": name,
            "definition": (f"The observation's overall category is '{name}'. "
                           f"{PLACEHOLDER_TEXT}"),
        })
        mapping[str(value)] = name
    return classes, mapping, generic


def _leaky_column(dataset: pd.DataFrame, benchmark: pd.DataFrame,
                  dataset_id: str | None, benchmark_id: str | None,
                  label_values: pd.Series) -> str | None:
    """A dataset column that copies the mapped human label.

    Duplicate IDs are tolerated: the label series is de-duplicated before the
    join, so a messier source file cannot crash the Review screen.
    """
    if not dataset_id or not benchmark_id:
        return None
    from general_agent.io import normalize_ids

    labels = pd.DataFrame({
        "__id": normalize_ids(benchmark[benchmark_id]).to_numpy(),
        "__label": benchmark[label_values.name].astype("string").str.strip().to_numpy(),
    }).dropna(subset=["__id"]).drop_duplicates(subset=["__id"]).set_index("__id")["__label"]
    left = dataset.copy()
    left["__id"] = normalize_ids(dataset[dataset_id])
    merged = left.merge(labels, left_on="__id", right_index=True, how="inner")
    if merged.empty:
        return None
    for column in dataset.columns:
        if column == dataset_id:
            continue
        values = merged[column].astype("string").str.strip()
        reference = merged["__label"]
        comparable = values.notna() & reference.notna()
        if comparable.sum() == 0:
            continue
        rate = float((values[comparable] == reference[comparable]).mean())
        if rate >= 0.98 and values[comparable].nunique() > 1:
            return column
    return None


def build_proposal(dataset_meta: dict, benchmark_meta: dict, *,
                   seed: int = 20261116) -> Proposal:
    dataset = dataset_meta["frame"]
    benchmark = benchmark_meta["frame"]
    proposal = Proposal(seed=seed, report_title="")
    proposal.problems = []

    discovery = discover.propose(dataset, benchmark, EMPTY_DATASET_CFG)
    mapping = discovery.mapping
    proposal.dataset_id_column = mapping.get("id_column")
    proposal.text_column = mapping.get("text_column")
    proposal.benchmark_id_column = mapping.get("id_column_benchmark")
    proposal.label_column = mapping.get("label_column")

    # --- columns ---------------------------------------------------------
    if not proposal.dataset_id_column:
        proposal.problems.append(Problem(
            "dataset_id_column", "blocking",
            "No column in the observation file looks like a unique ID.",
            "Choose an ID column, or tick 'create an ID from row numbers'."))
    if not proposal.text_column:
        proposal.problems.append(Problem(
            "text_column", "blocking",
            "No column in the observation file looks like free text.",
            "Choose the text column on this screen."))
    if not proposal.benchmark_id_column:
        proposal.problems.append(Problem(
            "benchmark_id_column", "blocking",
            "No column in the benchmark file looks like the matching ID.",
            "Choose the benchmark ID column on this screen."))
    if not proposal.label_column:
        proposal.problems.append(Problem(
            "label_column", "blocking",
            "No column in the benchmark file looks like the human label.",
            "Choose the human label column on this screen."))

    # --- categories and label mapping -----------------------------------
    values: list[str] = []
    if proposal.label_column and proposal.label_column in benchmark.columns:
        raw = benchmark[proposal.label_column].astype("string").str.strip()
        raw = raw[raw.notna() & (raw != "")]
        counts = raw.value_counts()
        values = [str(value) for value in counts.index]
        proposal.classes, proposal.label_map, generic = _propose_classes(values)
        if len(values) < 2:
            proposal.problems.append(Problem(
                "classes", "blocking",
                "The human label column has fewer than two distinct values.",
                "A classification task needs at least two categories."))
        elif generic:
            proposal.problems.append(Problem(
                "classes", "decision",
                "Some human labels are numeric or not usable as category names, so "
                "generic categories were proposed.",
                "Give each category a meaningful name and definition, and check the "
                "mapping below. Label polarity is never guessed."))
        else:
            proposal.problems.append(Problem(
                "classes", "decision",
                "Category names were taken from your labels and their definitions are "
                "placeholders.",
                "Review the category definitions; they are shown to the AI in the "
                "prompt."))

    # --- IDs and text quality -------------------------------------------
    from general_agent.io import normalize_ids
    stats = discovery.stats or {}
    alignment = stats.get("id_alignment") or {}
    if alignment.get("n_only_dataset") or alignment.get("n_only_benchmark"):
        proposal.problems.append(Problem(
            "benchmark.on_id_mismatch", "decision",
            f"IDs do not match exactly: {alignment.get('n_only_dataset', 0)} only in "
            f"the observations, {alignment.get('n_only_benchmark', 0)} only in the "
            "benchmark.",
            "Keep only matching IDs (counts are printed), or fix the files.",
            [{"action": "error"}, {"action": "inner_join"}]))
    duplicates = stats.get("duplicate_ids") or {}
    if duplicates:
        proposal.problems.append(Problem(
            "duplicate_ids", "blocking",
            f"Duplicate IDs found: {duplicates}.",
            "Remove or merge duplicate rows so each observation has one ID."))
    if proposal.text_column and proposal.text_column in dataset.columns:
        blanks = int(dataset[proposal.text_column].astype("string").str.strip()
                     .eq("").fillna(False).sum())
        if blanks:
            proposal.problems.append(Problem(
                "text_column", "blocking",
                f"{blanks} observation(s) have empty text.",
                "Remove or fill those rows before running."))
    if proposal.dataset_id_column and proposal.label_column:
        leaked = _leaky_column(dataset, benchmark, proposal.dataset_id_column,
                               proposal.benchmark_id_column,
                               benchmark[proposal.label_column])
        if leaked:
            proposal.problems.append(Problem(
                "leakage", "blocking",
                f"The observation file column '{leaked}' appears to copy the human "
                "label.",
                "Keep the observations and labels in separate files; the classifier "
                "must never see a label."))

    # --- model and budgets ----------------------------------------------
    providers = registry.enabled_providers()
    if providers:
        provider = providers[0]
        proposal.provider_id = provider["id"]
        models = provider.get("models", [])
        if models:
            proposal.model_name = models[0]["name"]
            model = models[0]
            ceiling = registry.max_calls_ceiling(model)
            rows = int(len(dataset))
            classification_calls = min(suggest_max_calls(rows, PILOT_N), ceiling)
            proposal.max_calls = classification_calls
            price = registry.price_block(model)
            proposal.max_usd = suggest_max_usd(rows, price)
            proposal.qualitative_enabled = True
            proposal.qualitative_sample_n = min(15, max(rows, 1))
            proposal.qualitative_control_n = min(5, max(rows, 1))
            proposal.qualitative_max_calls = (
                PILOT_N + 0 + proposal.qualitative_sample_n
                + proposal.qualitative_control_n + max(5, math.ceil(
                    (proposal.qualitative_sample_n + proposal.qualitative_control_n)
                    * 0.1)))
            proposal.qualitative_max_usd = max(proposal.max_usd, 0.10)
            _, present = registry.credential_status(model, provider)
            if not present:
                proposal.problems.append(Problem(
                    "credential", "warning",
                    "The chosen provider's credential is not set in this process.",
                    "Validation and sealing still work; classification needs it."))
    else:
        proposal.problems.append(Problem(
            "provider", "blocking",
            "No provider is enabled in configs/ui_providers.yaml.",
            "Enable a provider in that file, then reload."))

    # --- task text and notes --------------------------------------------
    text_column = proposal.text_column or "the observation"
    proposal.task_role = f"a classifier for {text_column}"
    proposal.decision_rule = ""
    proposal.report_title = ""
    proposal.notes = [
        "The classifier sees only the ID and the text of each observation.",
        "Human labels are sealed before classification and opened only after "
        "predictions are frozen.",
        "Qualitative explanations, when produced, are hypotheses, not confirmed "
        "causes.",
    ]
    return proposal

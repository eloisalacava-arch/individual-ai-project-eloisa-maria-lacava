"""Column discovery and the clarification policy.

Principle: resolve what is unambiguous, and ask only when a wrong guess would
silently corrupt the evaluation. Every automatic choice is recorded in
``decisions.json`` so the run is auditable and a re-run is non-interactive
(AGENTS.md: list every analysis choice you make).

What is never guessed:
  * label polarity - a numeric label without an explicit ``label_map`` is a
    question, not a heuristic. This is the single most damaging silent error in
    benchmark evaluation;
  * the class set - classes are declared in the config. Deriving them from the
    benchmark would leak the label set and its marginal distribution into the
    prompt;
  * which of two plausible text columns is the observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .config import class_names
from .io import normalize_ids

ID_NAME_HINTS = re.compile(r"(^id$|_id$|^uid$|^key$|^row_?id$|^doc_?id$|^record_?id$|_key$)")
TEXT_NAME_HINTS = re.compile(
    r"(text|body|review|sentence|comment|content|description|message|response|"
    r"article|headline|title|abstract|feedback|answer)"
)
LABEL_NAME_HINTS = re.compile(
    r"(label|sentiment|class|category|target|annotation|gold|truth|topic|rating|"
    r"polarity|stance|outcome)"
)

MAX_EXAMPLES = 3
EXAMPLE_WIDTH = 70


def _truncate(value: object, width: int = EXAMPLE_WIDTH) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def profile_column(name: str, series: pd.Series) -> dict:
    """Describe one column well enough to propose or reject a role for it."""
    n_rows = len(series)
    as_text = series.astype("string")
    stripped = as_text.str.strip()
    n_present = int(series.notna().sum())
    n_unique = int(stripped.nunique(dropna=True))
    n_blank = int(stripped.eq("").fillna(False).sum())
    lengths = stripped.str.len().dropna()
    unique_ratio = (n_unique / n_present) if n_present else 0.0

    return {
        "name": name,
        "dtype": str(series.dtype),
        "n_rows": n_rows,
        "n_present": n_present,
        "n_blank_or_null": n_blank + (n_rows - n_present),
        "n_unique": n_unique,
        "unique_ratio": round(unique_ratio, 4),
        "mean_chars": round(float(lengths.mean()), 1) if len(lengths) else 0.0,
        "median_chars": round(float(lengths.median()), 1) if len(lengths) else 0.0,
        "max_chars": int(lengths.max()) if len(lengths) else 0,
        "examples": [_truncate(value) for value in stripped.dropna().unique()[:MAX_EXAMPLES]],
    }


def profile_frame(frame: pd.DataFrame, role: str) -> list[dict]:
    return [profile_column(column, frame[column]) for column in frame.columns]


def _is_id_candidate(profile: dict) -> bool:
    if profile["n_rows"] == 0:
        return False
    return (
        profile["n_blank_or_null"] == 0
        and profile["unique_ratio"] >= 0.98
        and profile["median_chars"] <= 64
    )


def _is_text_candidate(profile: dict) -> bool:
    return profile["mean_chars"] >= 25 and profile["unique_ratio"] > 0.5


def _is_label_candidate(profile: dict) -> bool:
    # `<= 0.5` not `< 0.5`: a balanced binary label in a tiny audit sample has
    # exactly 2 unique values in 4 rows, and must still be recognised.
    return 2 <= profile["n_unique"] <= 40 and profile["unique_ratio"] <= 0.5


@dataclass
class Proposal:
    mapping: dict = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    profiles: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not self.questions

    def question(self, field_name: str, reason: str, options: list[dict],
                 how_to_resolve: str) -> None:
        self.questions.append({
            "field": field_name,
            "reason": reason,
            "options": options,
            "how_to_resolve": how_to_resolve,
        })


def _resolve_role(
    proposal: Proposal,
    role: str,
    profile_key: str,
    explicit: str | None,
    candidates: list[str],
    hint_pattern: re.Pattern,
    config_field: str,
    label: str,
) -> str | None:
    """Pick a column for a role, asking only when genuinely ambiguous."""
    names = [p["name"] for p in proposal.profiles[profile_key]]
    if explicit:
        if explicit in names:
            proposal.decisions.append(
                f"{role}: '{explicit}' (set explicitly in {config_field})"
            )
            return explicit
        proposal.question(
            config_field,
            f"{role} column '{explicit}' from the config is not in the {profile_key} columns {names}",
            [{"column": name} for name in names],
            f"correct {config_field} in the config",
        )
        return None

    if len(candidates) == 1:
        chosen = candidates[0]
        proposal.decisions.append(
            f"{role}: '{chosen}' (only candidate: {_candidate_reason(proposal, profile_key, chosen, label)})"
        )
        return chosen

    if len(candidates) > 1:
        hinted = [name for name in candidates if hint_pattern.search(name.lower())]
        if len(hinted) == 1:
            chosen = hinted[0]
            proposal.decisions.append(
                f"{role}: '{chosen}' (name matches the {label} convention among "
                f"candidates {candidates})"
            )
            return chosen
        proposal.question(
            config_field,
            f"{len(candidates)} plausible {role} columns: {candidates}; "
            "choosing the wrong one cannot be detected from the data alone",
            [{"column": name, **_candidate_summary(proposal, profile_key, name)}
             for name in candidates],
            f"set {config_field} in the config",
        )
        return None

    proposal.question(
        config_field,
        f"no column looks like a {role}; inspected {names}",
        [{"column": name, **_candidate_summary(proposal, profile_key, name)}
         for name in names],
        f"set {config_field} in the config",
    )
    return None


def _candidate_summary(proposal: Proposal, profile_key: str, name: str) -> dict:
    profile = next(p for p in proposal.profiles[profile_key] if p["name"] == name)
    return {
        "median_chars": profile["median_chars"],
        "n_unique": profile["n_unique"],
        "example": profile["examples"][0] if profile["examples"] else "",
    }


def _candidate_reason(proposal: Proposal, profile_key: str, name: str, label: str) -> str:
    profile = next(p for p in proposal.profiles[profile_key] if p["name"] == name)
    return (
        f"{label} of {profile['n_rows']} rows, {profile['n_unique']} unique, "
        f"median {profile['median_chars']} chars"
    )


def map_labels(series: pd.Series, label_map: dict | None,
               classes: list[str]) -> tuple[pd.Series, list[str]]:
    """Map raw labels onto canonical class names.

    Returns the mapped series (NA where unmapped/null) and the raw values that
    do not map. Unmapped values are always surfaced, never silently dropped.
    """
    raw = series.astype("string").str.strip()
    canonical = set(classes)
    if label_map:
        translation = {str(key).strip(): str(value) for key, value in label_map.items()}

        def translate(value):
            if pd.isna(value):
                return pd.NA
            key = str(value)
            if key in translation:
                return translation[key]
            # A value that already equals a declared class name is not a guess:
            # it is the canonical label spelled out. Anything else stays unmapped.
            return key if key in canonical else pd.NA

        mapped = raw.map(translate)
    else:
        mapped = raw.copy()

    unmapped = sorted(
        {
            str(value)
            for value, target in zip(raw, mapped)
            if pd.notna(value) and value != "" and (pd.isna(target) or str(target) not in canonical)
        }
    )
    mapped = mapped.astype("string")
    mapped = mapped.where(mapped.isin(canonical))
    return mapped, unmapped


def propose(dataset: pd.DataFrame, benchmark: pd.DataFrame, cfg: dict) -> Proposal:
    """Build the column mapping, or the questions needed to finish it."""
    classes = class_names(cfg)
    proposal = Proposal()
    proposal.profiles = {
        "dataset": profile_frame(dataset, "dataset"),
        "benchmark": profile_frame(benchmark, "benchmark"),
    }

    # ---- dataset: ID and text -------------------------------------------
    dataset_cfg = cfg["dataset"]
    dataset_ids = _resolve_role(
        proposal, "dataset ID", "dataset", dataset_cfg.get("id_column"),
        [p["name"] for p in proposal.profiles["dataset"] if _is_id_candidate(p)],
        ID_NAME_HINTS, "dataset.id_column", "unique-ID",
    )
    if dataset_ids:
        proposal.mapping["id_column"] = dataset_ids

    template = dataset_cfg.get("text_template")
    if template:
        missing = [
            name for name in re.findall(r"\{([^{}]+)\}", template)
            if name not in dataset.columns
        ]
        if missing:
            proposal.question(
                "dataset.text_template",
                f"template fields {missing} are not dataset columns {list(dataset.columns)}",
                [{"column": name} for name in dataset.columns],
                "fix dataset.text_template in the config",
            )
        else:
            proposal.decisions.append(
                f"dataset text: composed from template {template!r} "
                "(observation text is the joined fields, no other preprocessing)"
            )
            proposal.mapping["text_template"] = template
    else:
        dataset_text = _resolve_role(
            proposal, "dataset text", "dataset", dataset_cfg.get("text_column"),
            [p["name"] for p in proposal.profiles["dataset"] if _is_text_candidate(p)],
            TEXT_NAME_HINTS, "dataset.text_column", "free-text",
        )
        if dataset_text:
            proposal.mapping["text_column"] = dataset_text
    if "text_column" in proposal.mapping and "id_column" in proposal.mapping:
        if proposal.mapping["text_column"] == proposal.mapping["id_column"]:
            proposal.question(
                "dataset.text_column",
                "the same column was selected as both ID and text",
                [{"column": name} for name in dataset.columns],
                "set dataset.id_column and dataset.text_column explicitly",
            )

    # ---- benchmark: ID and label ----------------------------------------
    benchmark_cfg = cfg["benchmark"]
    benchmark_ids = _resolve_role(
        proposal, "benchmark ID", "benchmark", benchmark_cfg.get("id_column"),
        [p["name"] for p in proposal.profiles["benchmark"] if _is_id_candidate(p)],
        ID_NAME_HINTS, "benchmark.id_column", "unique-ID",
    )
    if benchmark_ids:
        proposal.mapping["id_column_benchmark"] = benchmark_ids

    label_candidates = [
        p["name"] for p in proposal.profiles["benchmark"]
        if _is_label_candidate(p) and p["name"] != benchmark_ids
    ]
    label_column = _resolve_role(
        proposal, "benchmark label", "benchmark", benchmark_cfg.get("label_column"),
        label_candidates, LABEL_NAME_HINTS, "benchmark.label_column", "categorical",
    )
    if label_column:
        proposal.mapping["label_column"] = label_column

    extra_columns = benchmark_cfg.get("extra_label_columns", [])
    for column in extra_columns:
        if column not in benchmark.columns:
            proposal.question(
                "benchmark.extra_label_columns",
                f"extra label column '{column}' is not in the benchmark columns {list(benchmark.columns)}",
                [{"column": name} for name in benchmark.columns],
                "fix benchmark.extra_label_columns in the config",
            )
    proposal.mapping["extra_label_columns"] = [
        column for column in extra_columns if column in benchmark.columns
    ]

    # ---- label mapping ---------------------------------------------------
    if label_column:
        _, unmapped_primary = map_labels(
            benchmark[label_column], benchmark_cfg.get("label_map"), classes
        )
        if unmapped_primary and benchmark_cfg["on_unknown_label"] == "error":
            if benchmark_cfg.get("label_map"):
                reason = (
                    f"label column '{label_column}' has values {unmapped_primary} "
                    "that benchmark.label_map does not map onto the declared classes "
                    f"{classes}"
                )
            else:
                reason = (
                    f"label column '{label_column}' has values {unmapped_primary} "
                    f"which are not declared class names {classes}. Label polarity is "
                    "never inferred; declare benchmark.label_map"
                )
            proposal.question(
                "benchmark.label_map", reason,
                [{"value": value} for value in unmapped_primary],
                "set benchmark.label_map (raw value -> declared class name)",
            )
        elif unmapped_primary:
            proposal.decisions.append(
                f"benchmark label: {len(unmapped_primary)} distinct unclassifiable value(s) "
                f"{unmapped_primary} will be excluded with counts and reason "
                "(benchmark.on_unknown_label = exclude)"
            )
        else:
            proposal.decisions.append(
                f"benchmark label: '{label_column}' maps onto all declared classes "
                f"{classes} (label_map {'applied' if benchmark_cfg.get('label_map') else 'not needed'})"
            )

        if label_candidates and len(label_candidates) > 1 and not benchmark_cfg.get("label_column"):
            proposal.decisions.append(
                f"benchmark extra categorical columns seen but not used: "
                f"{[c for c in label_candidates if c != label_column]}"
            )

    # ---- ID alignment ---------------------------------------------------
    if "id_column" in proposal.mapping and "id_column_benchmark" in proposal.mapping:
        dataset_id_values = set(normalize_ids(dataset[proposal.mapping["id_column"]]).dropna())
        benchmark_id_values = set(
            normalize_ids(benchmark[proposal.mapping["id_column_benchmark"]]).dropna()
        )
        only_dataset = sorted(dataset_id_values - benchmark_id_values)
        only_benchmark = sorted(benchmark_id_values - dataset_id_values)
        proposal.stats["id_alignment"] = {
            "n_dataset_ids": len(dataset_id_values),
            "n_benchmark_ids": len(benchmark_id_values),
            "n_matched": len(dataset_id_values & benchmark_id_values),
            "n_only_dataset": len(only_dataset),
            "n_only_benchmark": len(only_benchmark),
            "only_dataset_examples": only_dataset[:5],
            "only_benchmark_examples": only_benchmark[:5],
        }
        if (only_dataset or only_benchmark) and benchmark_cfg["on_id_mismatch"] == "error":
            proposal.question(
                "benchmark.on_id_mismatch",
                f"ID sets do not match: {len(only_dataset)} dataset ID(s) missing from the "
                f"benchmark and {len(only_benchmark)} benchmark ID(s) missing from the dataset "
                f"(examples: dataset-only {only_dataset[:3]}, benchmark-only {only_benchmark[:3]})",
                [{"action": "error"}, {"action": "inner_join"}],
                "set benchmark.on_id_mismatch to inner_join to keep the intersection "
                "(dropped rows are printed with counts and reasons)",
            )
        elif only_dataset or only_benchmark:
            proposal.decisions.append(
                f"ID sets differ; inner join approved by config: "
                f"{len(only_dataset)} dataset-only and {len(only_benchmark)} benchmark-only "
                "rows will be dropped with counts and reasons"
            )

    # ---- duplicate IDs ---------------------------------------------------
    duplicates = {}
    for role, frame, column in (
        ("dataset", dataset, proposal.mapping.get("id_column")),
        ("benchmark", benchmark, proposal.mapping.get("id_column_benchmark")),
    ):
        if column and column in frame.columns:
            ids = normalize_ids(frame[column])
            duplicate_count = int(ids.duplicated(keep="first").sum())
            if duplicate_count:
                duplicates[role] = duplicate_count
    if duplicates:
        proposal.stats["duplicate_ids"] = duplicates
        if not benchmark_cfg.get("allow_duplicate_ids"):
            proposal.question(
                "benchmark.allow_duplicate_ids",
                f"duplicate IDs found: {duplicates}. A many-to-many join would silently "
                "multiply rows in the evaluation",
                [{"action": "deduplicate the source file"}, {"action": "allow_duplicate_ids"}],
                "deduplicate the source, or set benchmark.allow_duplicate_ids to true "
                "only if duplicates are genuinely identical observations",
            )
        else:
            proposal.decisions.append(
                f"duplicate IDs present ({duplicates}) and allowed by config; "
                "the join stage must verify duplicates are identical rows"
            )

    return proposal


def observation_text(frame: pd.DataFrame, mapping: dict) -> pd.Series:
    """The observation text the classifier will see: one column, or a template.

    No normalisation happens here: no stripping, lowercasing, HTML decoding, or
    truncation. Any preprocessing changes what the model sees relative to what
    the human saw, so it must be an explicit, logged decision rather than a
    tidy-up hidden in a helper.
    """
    template = mapping.get("text_template")
    if template:
        def render(row: pd.Series) -> str:
            fields = {
                name: ("" if pd.isna(row[name]) else str(row[name]))
                for name in row.index
            }
            return template.format(**fields)

        return frame.apply(render, axis=1).astype("string")
    return frame[mapping["text_column"]].astype("string")

def redact_benchmark_examples(profiles: list[dict]) -> list[dict]:
    """Drop example cell values for benchmark columns that are not ID/label candidates.

    ``decisions.json`` is written outside the sealed directory, so it must not
    become a copy of benchmark metadata. Names, dtypes, and counts are kept
    because they are what a human needs to answer a clarification question.
    """
    redacted = []
    for profile in profiles:
        entry = dict(profile)
        if not (_is_id_candidate(profile) or _is_label_candidate(profile)):
            entry["examples"] = []
        redacted.append(entry)
    return redacted

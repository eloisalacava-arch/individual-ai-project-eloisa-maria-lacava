# MISTAKES.md — implementation mistakes and methodological decisions

**Project:** general-agent (Business Analytics expansion)
**Compiled:** 2026-11-16
**Scope:** phases 0–6: configuration, audit, blind/seal, pilot, classification,
evaluation, disagreement analysis, cross-dataset comparison.

**Evidence rule.** Every entry names how the problem was detected and what file
or test now guards against it. Where an error was caught before any number was
reported, that is stated; those cases have no retracted value to preserve.

Status categories:

| Category | Meaning | Count |
|---|---|---|
| **Implementation mistake** | code produced a wrong artefact or crashed | 23 |
| **Methodological decision** | a defensible choice that shapes the results | 12 |
| **Unresolved limitation** | a constraint that cannot be fixed with code | 8 |

---

## Part A — Implementation mistakes

| ID | Phase | Mistake | Detected by | Status |
|---|---|---|---|---|
| M1 | 1 | Label-candidate threshold rejected a valid binary label | unit test | fixed |
| M2 | 2 | `label_map` broke when some values were already canonical | real config (IMDb) | fixed |
| M3 | 1 | Configure printed every default as "applied" | user-visible output | fixed |
| M4 | 2 | Fetch summary reported 3,450 duplicate sentences instead of 5 | real run output | fixed |
| M5 | 2 | Sealed rows followed benchmark order, not blind order | unit test | fixed |
| M6 | 2 | `insert("id")` crashed because the benchmark already had `id` | real run (PhraseBank) | fixed |
| M7 | 3 | Parser rejected `ANSWER: "positive".` | unit test | fixed |
| M8 | 1 | `decisions.json` copied benchmark metadata values outside the seal | poison test | fixed |
| M9 | 1–2 | Re-running configure/audit wiped later stage records | regression test | fixed |
| M10 | 3 | Approval did not bind the pilot file hash | code review | fixed |
| M11 | 3 | Prompt guard false-positived on observations containing `<<<` | unit test | fixed |
| M12 | 3 | Pilot re-run guard could raise on an empty predictions file | code review | fixed |
| M13 | 3 | Call budget was counted per log file, not per run | pre-pilot arithmetic | fixed |
| M14 | 3 | Configured price was stale (0.038 vs 0.036) | model catalog check | fixed |
| M15 | 3 | Run summary printed the start-of-run call split | real run output | fixed |
| M16 | 3 | Cost totals reset to zero on a no-op resume | manifest refresh | fixed |
| M17 | 3 | Pilot cost was absent from the totals | cost reconciliation | fixed |
| M18 | 5 | Evaluation read the full 3,453-row source, not the sealed 300 | real evaluation run | fixed |
| M19 | 5 | Prediction and reference `label` columns collided in the merge | real evaluation run | fixed |
| M20 | 5 | Weighted kappa applied agreement weights as disagreement | unit-test design | fixed before use |
| M21 | 5 | F1 returned NaN when precision was zero | unit test | fixed |
| M22 | 6 | Comparison report hard-coded dataset names in generic code | genericity test | fixed |
| M23 | 6 | Disagreement stage hit the same `label` column collision | real run (PhraseBank) | fixed |

### M1 — Label-candidate threshold
A balanced binary label in a four-row sample has a unique ratio of exactly 0.5.
The predicate used `< 0.5`, so numeric label columns were never offered as
candidates and discovery could not even ask about them. Changed to `<= 0.5`.
Guard: `test_discover.py::test_numeric_labels_without_a_map_are_a_question`.

### M2 — `label_map` versus already-canonical values
One map is applied to the primary and every extra reference. For IMDb the
primary (`positive`/`negative`) was already canonical while the secondary was
`0`/`1`, so the map marked the primary unmapped and produced a spurious
clarification question. A value that already equals a declared class name now
passes through; anything else still requires an explicit map, so polarity is
never inferred. Guard: `test_label_map_passes_through_values_that_are_already_canonical`.

### M3 — Defaults printed as explicit choices
`configure` printed the full default table under "defaults applied", including
settings the user had written explicitly. It now diffs against the raw YAML and
prints only inherited keys. This is the stale-summary failure the parent project
records as D1; here it was caught in review, not by a crash.

### M4 — Fetch duplicate count
`duplicate sentences: 3450` while the data was correct (5 duplicates). The
comprehension unpacked `(sentence, label)` as `(_, text)`, so it counted labels.
The written files were unaffected; the console summary was wrong. Fixed and
re-run. Guard: duplicate count recomputed from the parsed pairs in the report.

### M5 — Sealed row order
`sealed/benchmark.csv` was written in benchmark order while `blind.csv` followed
the seeded shuffle, so row *i* of one did not correspond to row *i* of the
other. The join is by ID, so metrics were unaffected, but manual review was
misleading. Sealed rows now follow the blind order. Guard:
`test_blind.py::test_sealed_file_is_separate_and_restricted_to_the_sample`.

### M6 — `insert("id")` collision
The PhraseBank benchmark already has an `id` column, so inserting a second one
raised `ValueError: cannot insert id, already exists` during the first real
blind run. Changed to assignment plus column selection. Guard: the real
PhraseBank blind run.

### M7 — Parser punctuation
`ANSWER: "positive".` left a leading quote after the period was stripped,
producing `parse_unknown_class`. Normalisation now strips quotes, then periods,
then quotes again. Guard: `test_parsing.py::test_trailing_period_and_quotes_are_ignored`.

### M8 — Benchmark values in `decisions.json`
The audit stored example values for every profiled column, including
benchmark metadata. The poison-marker test for the classifier failed because the
marker appeared in `decisions.json` — outside the sealed directory. Example
values are now redacted for benchmark columns that are not ID/label candidates.
Guard: `test_leakage.py::test_poison_marker_never_leaves_the_sealed_file`.

### M9 — Manifest stages overwritten
`configure` and `audit` rebuilt `manifest.json` from scratch, discarding stage
sections written by later stages. Re-running an early stage silently erased
provenance for the blind/seal and classification stages. Both now merge the
`stages` mapping and keep the audit's top-level totals. Guard:
`test_blind.py::test_rerunning_audit_or_configure_preserves_later_stage_records`.
Note: this is the same class of bug as the parent project's D2 (stale artefact
beside the real one), caught here by a regression test rather than by a reader.

### M10 — Approval not bound to the pilot file
The approval stored the model and prompt-template hash but not the pilot
predictions hash, so replacing the pilot file did not invalidate approval. The
approval now records the pilot SHA-256 and stage 04 refuses on mismatch. Guard:
`test_classify.py::test_full_run_refuses_when_the_pilot_file_changed_after_approval`.

### M11 — Prompt guard false positive
The render guard counted occurrences of the observation in the prompt, so an
observation that literally contained `<<<` or `>>>` failed validation and would
have blocked a legitimate row. Validation now anchors on the template's own
delimiters and uses the last closer. Guard:
`test_prompts.py::test_observation_containing_delimiters_still_validates`.

### M12 — Pilot guard on an empty file
The "already holds results" check indexed `existing[0]` and could raise
`IndexError` on an empty predictions file. Replaced with a status-based check.

### M13 — Budget counted per file
`max_calls` was enforced separately for the pilot log and the full-run log, so a
config saying 300 allowed 5 pilot calls plus 300 full calls. The cap now counts
both logs. This was caught by the pre-flight arithmetic check the user
requested, before any pilot call. Guard:
`test_classify.py::test_pilot_calls_count_toward_the_total_budget`.

### M14 — Stale price
The config carried 0.038 USD per million input tokens; the model catalog
reported 0.036. Corrected before the pilot. The manifest records the price used.

### M15 — Start-of-run split in the summary
The full run printed `pilot 5 + full 0` and wrote the same split into the
manifest, because the split was captured before the loop. Recomputed from the
logs at the end. Detected by reading the real run output; the manifest was
refreshed with a zero-call resume.

### M16 — Cost totals zeroed on resume
Cost was accumulated in memory, so a resume with no pending rows overwrote the
manifest totals with zero. Totals are now recomputed from the attempt logs.
Guard: `test_classify.py::test_noop_resume_does_not_reset_cost_totals`.

### M17 — Pilot cost missing from totals
The pilot ran before per-call cost capture existed, so its records had no
`cost_usd` and the total undercounted. Instead of editing frozen logs, the
runner derives cost from the preserved raw `step_finish` output. Guard:
`test_classify.py::test_totals_are_derived_from_raw_output_when_cost_was_not_logged`.

### M18 — Evaluation read the source benchmark
`load_benchmark` read `cfg["benchmark"]["path"]` (3,453 rows) instead of
`sealed/benchmark.csv` (300). The join check reported 3,153 benchmark rows
without a prediction, which exposed it. Evaluation now reads only the sealed
file; the source benchmark is never opened during scoring. Guard: the real
evaluation run and `test_evaluate.py`'s sealed-file fixture.

### M19 — Column collision in the join
The benchmark's label column is also called `label` in the PhraseBank config,
which is the prediction column name. pandas would suffix `_x`/`_y`, and the
rename to `reference` silently found nothing (`KeyError`). The prediction column
is renamed before the merge. Guard: real evaluation run.

### M20 — Weighted kappa sign inversion
The first implementation subtracted a disagreement term computed from
*agreement* weights, which assigns kappa = 1 to maximally disagreeing data. It
was caught while writing the unit tests, before any reported number. The
standard form `(P_o^w - P_e^w) / (1 - P_e^w)` is now used. Guard:
`test_metrics.py::test_weighted_kappa_is_not_inverted_for_opposite_errors`.

### M21 — F1 with zero precision
`if precision and recall` treated a zero precision as "undefined", so F1 became
NaN instead of 0. Fixed; the term is now 0 when either is 0 and both are
defined. Guard: `test_metrics.py::test_f1_is_zero_when_precision_is_zero_not_nan`.

### M23 — The same column collision, twice
The PhraseBank config names its reference column `label`, which is also the
prediction column name. Stage 05 was fixed by renaming the prediction before the
merge, but stage 06 repeated the merge independently and crashed with
`KeyError: 'label'` on the first real PhraseBank run. Stage 06 now renames the
prediction first and asserts the required columns after the merge. The
recurrence is itself the lesson: a fix applied at one call site does not protect
another, so the guard is a post-merge column assertion rather than a comment.

### M22 — Dataset names in generic comparison code
The generated report hard-coded two bullets naming the case studies. The
comparison module is now dataset-agnostic; study-specific limitations come from
each config's `report.limitations`. Guard:
`test_genericity.py::test_no_dataset_name_in_evaluation_code`.

---

## Part B — Methodological decisions

These are choices, not errors. They are listed so a reader can disagree
knowingly. The frozen record is `SPEC.md` §7.

1. **Seeded, unstratified sampling** (300 of 3,453; seed 20261116). Stratifying
   would fix the class distribution and inflate kappa, which depends on the
   marginals. Preserving the natural prevalence was preferred over a balanced
   design.
2. **Classes declared, never inferred** from the benchmark, to avoid leaking the
   label set or its distribution into the prompt.
3. **Label polarity never inferred.** A numeric reference requires an explicit
   `label_map`.
4. **Blind input is exactly `(id, text)`**, written to its own directory; the
   sealed benchmark lives elsewhere and is `chmod 000` during classification.
5. **Pilot gate of five observations**, bound to model, prompt-template hash,
   and pilot-file hash.
6. **Temperature is null** for both studies and recorded as `not_requested`;
   no claim of controlled sampling.
7. **Budget counts pilot and full calls together**, from the attempt logs.
8. **Predictions are frozen (copied + hashed) before the benchmark is opened**,
   and the scored copy is the frozen one.
9. **Evaluation reads only the sealed reference**, never the full source.
10. **Accuracy and recall use Wilson intervals; kappa, balanced accuracy,
    precision, and F1 use seeded percentile bootstrap**, because no simple
    binomial model applies to those quantities.
11. **Disagreements are reported as unconfirmed**; zero confirmed model errors
    are estimable without per-annotator labels or adjudication.
12. **The length comparison is exploratory**: one pre-specified feature
    (observation word count), seeded permutation test, no multiplicity
    adjustment, no causal reading.

---

## Part C — Unresolved limitations

1. **No human–human agreement, and no intra-annotator check, for either study.**
   The PhraseBank release ships majority labels only; the IMDb manual annotation
   is a single annotator, and the codebook's planned intra-annotator
   re-annotation of ~20 reviews (`../codebook.md` §7) was not carried out
   (`../mistakes.md` U1).
2. **The PhraseBank sample is the ≥75% agreement subset**, so it excludes
   low-agreement sentences and plausibly overstates agreement with the majority
   label relative to the full corpus.
3. **Kappa is prevalence-sensitive**; cross-study comparison without marginals
   is not meaningful.
4. **Reference labels are not adjudicated truth**, so disagreements cannot be
   separated from reference ambiguity.
5. **Small negative class in the PhraseBank sample** (n=38) gives wide per-class
   intervals.
6. **Temperature and provider routing are uncontrolled**, so run-to-run stability
   is only weakly evidenced (5 of 5 pilot rows reproduced).
7. **No per-annotator data**, so the published item-level agreement cannot be
   converted into a comparable kappa.
8. **Pretraining contamination cannot be excluded** for either domain.

---

## Part D — What this record is for

The parent project's `mistakes.md` concluded that fluent output is not verified
output, that agents document problems without fixing them, and that hard-coded
numbers are a governance failure. The same failure modes appeared here:

- **M4** produced a confident, wrong number in a console summary while the data
  were correct.
- **M15** and **M16** left stale totals inside the artefact a reader would trust.
- **M18** would have scored 300 predictions against a 3,453-row file and
  reported a plausible, wrong denominator; the join check caught it.
- **M19** would have merged on suffixed columns without an error and produced
  nonsense metrics if the rename had silently succeeded with `_x`.
- **M20** would have reported kappa = 1 for maximally disagreeing data.

The durable defences are structural, not promises: reconciliation checks,
freeze-before-read, sealed-only evaluation, and tests that fail closed.

*End of document.*
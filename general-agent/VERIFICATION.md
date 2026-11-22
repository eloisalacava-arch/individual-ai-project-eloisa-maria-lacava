# VERIFICATION.md — verification record

**Project:** general-agent (individual project), plus the earlier IMDb group
project where it provides independent evidence.
**Rule:** no live model calls, no statistic recomputed here; every claim names a
file or a test. Two kinds of check are distinguished throughout.

- **Independent** — recomputed by separate code written without reusing the
  project's analysis code, and compared only afterwards. In this repository that
  is the earlier **group** project's verification pass
  (`../results/verification_findings.md`, and `../mistakes.md` Part F for the
  McNemar correction). The individual agent has **no** independent external
  verification.
- **Same-system** — checked by the agent's own tests and provenance, written by
  the same development process. These are strong regression guards, not
  independent confirmation.

## 1. Claim to evidence

| Claim | Kind | Evidence |
|---|---|---|
| The classifier reads only `id` and `text`; a poison marker never leaves the sealed file | same-system | `tests/test_leakage.py` (poison marker, import-boundary AST check, file-access spy) |
| The sealed benchmark is separate and restricted to the sample; sampling is seeded and unstratified | same-system | `tests/test_blind.py` |
| Predictions are frozen and hashed before the sealed reference is opened | same-system | `tests/test_evaluate.py::test_predictions_are_frozen_before_the_benchmark_is_read`; `runs/*/results/evaluation_log.txt` `[PASS] benchmark_opened_after_freeze`; `runs/*/results/t01_join_and_sample.csv` |
| The sealed directory is locked during every model call | same-system | `tests/test_classify.py::test_sealed_directory_is_locked_during_calls_and_unlocked_after`; `tests/test_ui_pipeline_guards.py::test_seal_precedes_every_model_call` |
| Accuracy and Cohen's kappa reproduce from the frozen predictions | same-system, independent code path | `tests/test_reproduce_case_studies.py` vs `runs/*/results/t02_accuracy.csv`, `t03_kappa.csv`, `t13_reference_agreement.csv` |
| IMDb kappa targets 0.9399 / 0.8899 / 0.850 | **independent** (group) and re-checked same-system | `../results/verification_findings.md`; `../results/agent_report.md`; asserted in `tests/test_reproduce_case_studies.py` |
| Calls stay within the configured caps; resume and fatal errors are handled | same-system | `tests/test_classify.py`; `tests/test_pipeline.py`; `tests/test_manifest_accounting.py` |
| The real three-step UI run stayed within its authorized caps | same-system | `tests/test_acceptance_simple_interface_run.py`; `runs/simple_interface_test/ui_authorization.json`, `manifest.json` |
| The report composer presents; it never recomputes a statistic | same-system | `tests/test_report_boundary.py` |
| No dataset name appears in the evaluation, qualitative or report code | same-system | `tests/test_genericity.py` |
| The shipped case-study runs and `comparison/` are read-only during a new run | same-system | `tests/test_ui_case_studies.py`; `tests/test_acceptance_new_user.py` |
| M7 (McNemar cells) was found and corrected | **independent** (group) | `../mistakes.md` Part F |

## 2. Implemented vs not implemented statistical methods

Implemented and exercised (`runs/<run_id>/results/t01`–`t16`): confusion matrix,
accuracy (Wilson CI), balanced accuracy and per-class precision/recall/F1,
Cohen's kappa (unweighted; quadratic when ordered) with bootstrap CIs, the
reference-agreement matrix, disagreement tables, the exploratory length test,
and the v0.8 subgroup contrasts (Newcombe risk differences, Fisher exact tests,
Benjamini–Hochberg q across the pre-specified flags only, three-way
decomposition, reference-dispute rate).

**Documented but NOT implemented** (see `SPEC.md` §4/§8): McNemar exact test,
Holm-adjusted p-values, minimum detectable differences, and dual
abstention-policy tables. These must not be cited as delivered checks.

A **subgroup n-sum reconciliation is deliberately not asserted**: the difficulty
flags are independent multi-label judgments, so their per-subgroup sample sizes
overlap and must not be required to sum to the analysed total (`SPEC.md` §4).

**Not delivered because they are human-only, not code:** the manual annotation
has a single annotator, and the codebook's planned intra-annotator re-annotation
(`../codebook.md` §7) was not carried out, so neither intra- nor inter-annotator
reliability is available (`../mistakes.md` U1, U2). No reliability number is
reported, so there is nothing here to verify.

## 3. Manifest call accounting

`tests/test_manifest_accounting.py` asserts every run: no stage's
`calls_used_total` exceeds its `max_calls`, and no stage's `actual_usd` exceeds
the run's `max_usd`. There are **no budget violations**.

Two **legacy schema inconsistencies** are preserved and documented, not
rewritten: the `03_pilot` records for `imdb_case_study` and
`phrasebank_75agree` show `calls_pilot = 0`, `calls_full = 0` but
`calls_used_total = 5`, because they were written by an older code path. Later
runs (`simple_interface_test`, `study`) and later stages record a consistent
split. The test fails if any *new* mismatch appears.

Also noted, not a violation: the phrasebank manifest's top-level `budget`
records `max_calls = 305` (written at configure) while the operative stage cap in
`04_classify` is `320`; the stage cap is the one enforced, and 305 calls were
used.

## 4. Configuration history

`config_history.json` records, for every run, the configuration hashes observed
in order, each copied from the file that recorded it; original hashes are never
overwritten. Summary:

| Run | Configure hash | Later recorded hashes | Current |
|---|---|---|---|
| `imdb_case_study` | `4f2000d3…` (2026-11-16) | `c82f6085…` (qualitative, 11-17), `ddea97eb…` (evaluate, 11-19), `c850923f…` (report, 11-19) | `c850923f…` |
| `phrasebank_75agree` | `a24767e8…` (2026-11-16) | `7bf1eaff…` (qualitative 11-17; evaluate 11-16; report 11-19) | `7bf1eaff…` |
| `simple_interface_test` | `14d06bc9…` | same at configure/evaluate/report | `14d06bc9…` |
| `study` | `3876d2dd…` | same at configure/evaluate/report | `3876d2dd…` |

The IMDb predictions were **imported** from
`data/derived/workflow_13_predictions_openrouter.csv` on 2026-11-16, before every
later configuration change, so the current config did not generate them. The
later changes to `configs/imdb_case_study.yaml` were analysis/report-only
(subgroup keys, hypothesis text, `reference_labels`). PhraseBank's predictions
were produced by a live `04_classify`; the legacy manifest records no config hash
for that stage, so its exact classification-time hash is not recoverable.

## 5. Remaining discrepancies (transparent)

1. **Config-hash drift.** For the case studies the manifest's original
   `config.sha256`, the evaluation-time hash, and the current config differ. This
   is recorded in `config_history.json`; it reflects analysis-only edits and does
   not affect predictions or result tables.
2. **Qualitative approval binding.** `runs/imdb_case_study/qualitative/approval.json`
   still records `study_config_sha256 = c82f6085…` (the original), while the
   settings sidecar was re-bound to the current hash so the report composer can
   read it. The historical value is preserved and not rewritten.
3. **Qualitative enabled but not run** in `simple_interface_test`: the
   authorization enabled it, but the run had 0 disagreements, so no explanations
   were produced. This is consistent, not a failure.
4. **Legacy manifest schema.** Stages `05_evaluate`/`06_disagreements` do not
   record a `config_sha256`; the hash is in `results/run_metadata.json` instead.
5. **No independent verification of the individual project.** All general-agent
   checks above are same-system. The only independent evidence is the earlier
   group project's verification pass, which covers the IMDb group numbers, not
   the PhraseBank run, the qualitative stage, or the comparison.

*Reviewed 2026-11-19. No prediction file, result table, or qualitative
explanation was changed or regenerated to produce this record.*

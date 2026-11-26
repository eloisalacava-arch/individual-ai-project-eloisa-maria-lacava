# VERIFICATION.md — verification record

**Project:** general-agent (individual project), plus the earlier IMDb group
project where it provides independent evidence.
**Rule:** no live model calls; for §§1–7 no statistic is recomputed and every
claim names a file or a test. Two kinds of check are distinguished throughout
(independent and same-system). §8, added 2026-11-26, is different in kind: it
records **AI-assisted arithmetic** and **manual** interface checks, each labelled
there, and it is **not** independent verification.

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
| The interface corrections recorded in `SPEC.md` v0.5 (blind-stage ID-intersection and label-exclusion policies, the real failed-prediction count, the single-study heading) are held by regression tests | same-system | `tests/test_blind.py::test_inner_join_policy_is_materialised_by_blind`, `::test_unknown_labels_are_excluded_by_blind_with_counts`; `tests/test_evaluate.py::test_evaluation_log_reports_the_real_failure_count`; `tests/test_report_reuse.py::test_study_report_matches_the_single_study_comparison_report` |
| The report/PDF defects found in the manual acceptance run (binary misdescription, a spurious "weakest category", blank reference metadata, executive-summary heading spacing) are held by regression tests | same-system | `tests/test_report_defects.py` |
| The six academic-report corrections (multiplicity wording, prediction-bias direction, share denominator, the missing-statistics claim, wide-table/PDF layout, the reproduction command) are held by regression tests | same-system | `tests/test_report_fixes.py`; `tests/test_acceptance_simple_interface_run.py::test_evaluation_and_study_report_completed` |
| The shipped case-study runs and `comparison/` are read-only during a new run | same-system | `tests/test_ui_case_studies.py`; `tests/test_acceptance_new_user.py` |
| GP-M7 (McNemar cells) was found and corrected | **independent** (group) | `../mistakes.md` Part F |

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
6. **Open `t08` severity-label inconsistency.** For the binary IMDb run,
   `runs/imdb_case_study/results/t08_disagreements.csv` labels all six
   disagreements `adjacent`, while `t07_ordinal_errors.csv` and
   `comparison/tables/cross_dataset_severity.csv` label them `opposite`. No metric
   or count changes and the frozen CSV is unedited; `t07` and the cross-dataset
   table are authoritative for severity. Recorded in
   `ERROR_ANALYSIS_SUBMISSION.md` §8 and `publication/README.md`. No committed
   test fails on it, so it is **not** currently guarded.

## 6. Reproducing the same-system checks

The same-system checks re-run without a model call:

```bash
cd general-agent
uv run pytest tests/test_reproduce_case_studies.py tests/test_manifest_accounting.py tests/test_leakage.py
```

`tests/test_reproduce_case_studies.py` and
`tests/test_acceptance_simple_interface_run.py` call `pytest.skip` when their
committed run evidence (`runs/<run_id>/`, `runs/simple_interface_test/`) is
absent, so a green run says nothing about those claims unless the case-study
directories are present. `tests/test_manifest_accounting.py` does not skip: with
no manifests it simply finds no cap violation.

## 7. External installation and interface smoke test (not statistical verification)

An external tester independently followed the public GitHub README and reported:

- the repository structure was understandable;
- installation and Streamlit start-up succeeded;
- the interface displayed and navigated correctly.

The tester did **not** test classification, API calls, or statistical accuracy.

This is an external **installation and interface** check only. It does not
recompute or confirm any statistic, prediction, reference label, or model output,
so it is **not** independent statistical or model verification and does not
change the internal-versus-independent distinction above. The two clarity
suggestions it produced — installation-directory wording and a demonstration
dataset — are addressed in the READMEs; the demonstration data are the existing
synthetic files under `acceptance/`.

## 8. User-led interface review and AI-assisted comparison of results and reports (26 November 2026; not independent verification)

This section integrates `VERIFICATION_ADDENDUM.md`. It records a second, later
review: a **user-led** interface walk-through by the project author, combined with
**AI-assisted** arithmetic and prose-to-table comparison. Neither is independent
verification. Where §7 records an external tester's installation smoke test, this
section records the author's own review; the two do not overlap.

Each row labels who did the check. **Manual** = the project author operated the
interface, exported the reports and shared a screenshot. **AI-assisted** = an
assistant re-derived arithmetic from the committed tables, or read the exported
PDFs and compared them against those tables. Unlike §§1–7, this section contains
AI-assisted arithmetic, so it re-derives numbers rather than only naming them; the
source table is named for each row so the arithmetic can be checked by hand.

| Check | Method | Source | Finding |
|---|---|---|---|
| Interface startup | **Manual** (author launched Streamlit and shared a screenshot) | expected Upload, Review, Analyze & Download, Advanced mode, Case studies | Interface loaded; navigation appeared |
| Case-study access | **Manual** (author opened both existing case studies) | IMDb and Financial PhraseBank | Both displayed without a new classification |
| IMDb accuracy | **AI-assisted** arithmetic: 194/200; confusion-matrix diagonal 101+93 | `runs/imdb_case_study/results/t02_accuracy.csv`, `t04_confusion_matrix.csv` and the IMDb PDF | 97.00%, consistent |
| PhraseBank accuracy | **AI-assisted** arithmetic: 274/300; diagonal 37+171+66 | the PhraseBank `t02_accuracy.csv`, `t04_confusion_matrix.csv` and PDF | 91.33%, consistent |
| Sample accounting | **AI-assisted** comparison of analysed, dropped and excluded counts | both `t01_join_and_sample.csv` files | 200 and 300 analysed; zero dropped or excluded failed predictions |
| Cohen's kappa | **AI-assisted** comparison of PDF values with the committed tables (no fresh re-estimation) | both `t03_kappa.csv` files | Rounded values consistent: 0.940 and 0.841 |
| Disagreement structure | **AI-assisted** comparison of report summaries with the result tables | the study and cross-dataset reports | IMDb: 6/200; PhraseBank: 26/300 (23 adjacent, 3 opposite) |
| Report export | **Manual** export; **AI-assisted** inspection | IMDb PDF (10 pages), PhraseBank PDF (8 pages), cross-dataset PDF (6 pages) | All opened and contained tables, figures, interpretation and limitations |

**Errors and discrepancies identified.**

- **IMDb severity mismatch — reconfirmed, no new detail.** The six binary
  disagreement rows in `runs/imdb_case_study/results/t08_disagreements.csv` are
  labelled `adjacent`, while `t07_ordinal_errors.csv` and the cross-dataset
  severity table label all six `opposite`. §5.6 above already records this, and
  this review reached the same conclusion without adding anything new. The
  maximum-distance interpretation is appropriate for a binary classification; the
  inconsistency affects row-level severity labels only, not accuracy, kappa or the
  disagreement count. The issue remains unresolved and unguarded.
- **IMDb qualitative interpretation mismatch (new, §5 does not cover it).** The
  IMDb PDF (page 9) states that `Other` appeared in *both* the disagreements and
  the control. Its preceding table (page 8), and
  `runs/imdb_case_study/qualitative/tables/q07_reason_contrast.csv`, show `Other`
  at 0.0% of disagreements and 10.0% of controls — it appears only in the control.
  The narrative therefore does not match its own table. This was found by reading
  the generated prose against its supporting table rather than trusting the prose.
  It is not currently held by a regression test: the defects guarded by
  `tests/test_report_defects.py` are different ones.
- **Minor report formatting (new).** The PhraseBank PDF has an empty bullet in its
  study-description section (page 2), and the cross-dataset comparison PDF has a
  partially overlapping chart legend (page 6). Presentation only; no reported
  statistic is affected.

**Automated testing and limitations.** The project's recorded final offline suite
is reported at **353 passed, zero failed, zero skipped**. This figure is quoted
from the recorded run report and was **not** re-run for this review; it is not
re-derived here and no test-run report for it is committed in this repository. The
same-system tests of §1 remain the guards for the label-separation and
freeze-before-benchmark safeguards.

This review had limits: the project author did **not** carry out a fresh
synthetic-data upload-and-seal test, and did not manually inspect every prompt and
model request. No new paid classification calls, no full end-to-end rerun, no
independent label adjudication and no external statistical audit were performed.
Accuracy measures agreement with the chosen human reference, not proven
classification truth. The PDF/CSV checks were **AI-assisted**, not an unaided
independent verification by the author.

**Conclusion.** The checked numerical results match their underlying committed
tables, and the interface and exported PDF reports were accessible. The two IMDb
mismatches show that plausible agent-generated prose and row-level outputs still
need checking against the underlying evidence. A green test suite supports
internal consistency but does not replace critical human review or independent
statistical validation.

*Reviewed 2026-11-19. No prediction file, result table, or qualitative
explanation was changed or regenerated to produce this record.*

*§8 added 2026-11-26 from `VERIFICATION_ADDENDUM.md`. Documentation only; no
source code, model setting, prediction, label or statistical result table was
altered.*

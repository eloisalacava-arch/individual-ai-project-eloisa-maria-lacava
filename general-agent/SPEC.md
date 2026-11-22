# SPEC.md — Method record (v0.11)

**Project:** General-purpose AI classification and evaluation agent
**Status:** phases 0–9 implemented (engine, Streamlit interface, qualitative
disagreement analysis, per-study and comparison report composers, and
pre-specified subgroup contrasts). Both studies are classified, evaluated,
analysed for disagreement structure and subgroup contrasts, and compared in a
generated report. Packaging is not built.
**Rule:** this file is the frozen method record. Changing it is a version bump,
and the change is logged in the changelog at the bottom.

Status labels follow the parent project's convention:

- **[Verified]** — checked against a file or command output, cited here;
- **[External]** — from published documentation, not verifiable from local files;
- **[Decision]** — an analysis choice made deliberately.

---

## 1. Purpose and scope

A reusable agent that classifies observations with an AI model and compares
those predictions against a human benchmark, for any single-label
classification task with ≥2 classes. The IMDb project is case study 1; the
Financial PhraseBank is case study 2.

**In scope for v0.1:** single-label classification, binary or multiclass,
ordinal or nominal, CSV/Parquet/JSONL inputs, one primary human benchmark plus
optional extra reference columns.

**Out of scope for v0.1 (rejected explicitly, not half-supported):**
multi-label classification (`labels.mode` accepts only `single_label`),
sequence labelling, regression, and multi-annotator long-format files requiring
per-annotator agreement computation. The last is a real limitation: the agent
compares AI against a *consensus* label, it does not compute human–human kappa.

---

## 2. Model access and temperature **[Verified]**

Checked on this machine on 2026-11-16:

| Question | Finding | Evidence |
|---|---|---|
| Does `opencode run` accept `--temperature`? | **No.** | `opencode run --help` lists `--model`, `--agent`, `--format`, `--file`, `--title`, `--archived`, `--thinking`, `--auto`, `--continue`, `--session`, `--fork`, `--worktree`. No sampling parameter of any kind. |
| Can temperature be set at all? | **Yes, via configuration.** | OpenCode docs: `providers.<id>.body` and `providers.<id>.models.<model>.body` are "JSON fields merged into every request body". Temperature would be injected there. |
| Is the value honoured? | **Not guaranteed.** | Docs: model options are provider-specific and "may be ignored or rejected by another provider package". Model capability metadata declares temperature support but does not confirm per-call behaviour. |

**Consequence [Decision]:** `model.temperature` is stored as
`temperature_requested`, and the manifest records
`temperature_status: not_requested | requested_unverified`. The report must
never claim a controlled temperature on the strength of a config value. If the
value is null or unverifiable, run-to-run stability is measured rather than
assumed (the U5 gap in the parent project's `mistakes.md`).

This refines `src/workflow_00_model_and_prompt.md` §1, which recorded
temperature as "not settable through the access method". That was true of the
CLI flags actually used; it is not true of the access path as a whole.

---

## 3. Blinding and leakage control **[Decision]**

The classifier must not see any benchmark label, and the evaluation must not be
able to influence predictions. Five layers, in the order they will be built:

1. **Filesystem split (phase 2, built).** `02_blind` writes `blind/blind.csv`
   with exactly `(id, text)` and `sealed/benchmark.csv` separately; the
   classifier subprocess runs with `cwd` set to the blind directory, which is
   asserted to contain one file.
2. **Import boundary (phase 2, built).** `classify.py`, `prompts.py`,
   `parsing.py`, `sealing.py`, and `providers/` import no benchmark-side module.
   A test audits the import graph, so this cannot regress silently.
3. **Payload guard (phase 3, built).** Before every call, the rendered prompt
   must equal `build_prompt(cfg, text)` exactly, and the delimited observation
   must equal the blind row's text verbatim. Otherwise the call is refused.
4. **Poison test (phase 3, built).** A unique marker planted in a benchmark
   metadata column must never appear in a prompt, prediction file, log, or
   manifest. The test also checks `decisions.json`, which is why the audit
   redacts example values for benchmark columns that are not ID/label
   candidates.
5. **Directory lock (phase 3, built).** The sealed directory is `chmod 000`
   for the duration of a classification run and restored in a `finally` block.
   A same-user process cannot traverse it. This is a barrier, not a sandbox: it
   does not stop root and it does not stop a process that waits for the unlock.
6. **Freeze order (phase 5).** `predictions_sha256` is recorded before the
   benchmark is unsealed; evaluation aborts if the file changed afterwards.
   The hash is already recorded at stage 04; the enforcement arrives with the
   evaluation stage.

Layer 0, already implemented: the **audit** refuses a dataset that carries a
benchmark label column, by name or by near-perfect value equality with a
reference column. A combined "text + labels" file must be split before use.

**[Decision]** Class names are declared in the config, never inferred from the
benchmark. Inferring them would leak both the label set and its marginal
distribution into the prompt.

**[Decision]** Numeric label polarity is never inferred. Without an explicit
`benchmark.label_map`, a numeric label column produces a question.

**[Decision] The interface proposal step does not weaken the seal.** In Simple
mode the interface reads the uploaded benchmark's label column only to *propose*
category names and a value-to-category mapping, and to flag an observation column
that copies the label. Those proposals are shown to the human reviewer for
confirmation (`app/propose.py`); they are not sent to the classifier. The
classifier still reads only `blind/blind.csv` (`id`, `text`) plus the declared
class names and definitions that the confirmed config states, and the sealed
benchmark is `chmod 000` for the duration of classification. No row-level
benchmark label, count, or distribution enters a prompt, and the `label_map` is
applied to the reference only at evaluation, after predictions are frozen.

---

## 4. Evaluation method (phase 5)

Every number is written to `runs/<run_id>/results/*.csv`; the report will be
generated only from those files (parent AGENTS.md rule).

**Implemented and exercised** (`runs/<run_id>/results/t01`–`t16`):

- **Confusion matrix** (`t04`), rows = human reference, columns = AI; the
  orientation is stated in the table header, not implied.
- **Accuracy** (`t02`) with a Wilson score CI; **balanced accuracy** and
  **per-class precision/recall/F1** with support, so class imbalance cannot hide
  behind a single accuracy figure.
- **Cohen's kappa** (`t03`), unweighted; quadratic-weighted when
  `labels.ordered` is true. Both use a seeded percentile bootstrap CI.
- **Reference matrix** (`t13`): agreement and kappa for every pair among {AI,
  primary reference, extra references}. This is what reproduces the parent
  project's three 2×2 tables and structurally prevents the M6/M7
  cell-substitution bug.
- **Disagreement analysis** (`t08`, `t11`, `t12`, `t12b`): one row per
  disagreement labelled **unconfirmed**; reference→prediction pair counts; an
  exploratory seeded observation-length test.
- **Subgroup contrasts** (`t14`–`t16`, added v0.8): Newcombe hybrid-score
  risk-difference intervals, two-sided Fisher exact tests, and
  Benjamini–Hochberg q-values across the pre-specified flags only; a three-way
  AI / primary / secondary-reference decomposition; and reference-disagreement
  rate by subgroup.
- **Reconciliation**: `t01_join_and_sample.csv` records every row count from
  freeze to the analysed set, and the evaluation asserts
  `disagreement_count_reconciles` (disagreements = n − correct).
- **Abstentions/failures** are never imputed; failed predictions are excluded
  and counted in `t01`.
- **`not_estimable`** is a legal output for a degenerate cell (zero errors in a
  subgroup, a class with no support, complete separation).
- **No causal claims.** Subgroup contrasts are observational associations.

**Documented but NOT implemented** (these are not delivered checks and must not
be cited as if they were):

- **McNemar exact test** for paired binary comparisons, with discordant counts.
  No `t01`–`t16` table contains a McNemar statistic and no code computes one.
- **Holm-adjusted p-values.** `t14` reports raw Fisher p and Benjamini–Hochberg
  q only.
- **Minimum detectable differences.** No MDE is computed or printed; the agent
  avoids "no difference" phrasing rather than quoting power.
- **Both abstention policies as tables.** `evaluation.abstention_policy` is
  validated and mentioned in the evaluation log, but no table reports `exclude`
  and `count_as_error` side by side.
- **Subgroup n-sum reconciliation.** Per-subgroup n values are written but not
  asserted to sum to the analysed n. This is a deliberate non-check, not an
  omission: the difficulty flags (`sarcasm`, `mixed`, `ambiguous`, `complex`) are
  independent multi-label judgments (`../codebook.md` §4), so one row can carry
  more than one and the per-flag n values are not a partition of the sample.
  Requiring them to sum would be incorrect, so no such assertion is made.

---

## 5. Case study 1 — IMDb (binary, reused predictions)

Purpose: reproduce verified numbers with **zero new model calls**.

| Item | Value | Status |
|---|---|---|
| Dataset | `data/derived/workflow_classifier_input.csv` (200 rows, `review_id`, `text`) | **[Verified]** |
| Predictions | `data/derived/workflow_13_predictions_openrouter.csv` (200 rows, all `status=ok`, one model) | **[Verified]** at audit |
| Primary reference | `human_sentiment` from `data/annotations.csv` | **[Verified]** |
| Secondary reference | `imdb_label`, star-derived, 0=negative / 1=positive | **[Verified]** `data/README.md` §2 |
| Benchmark file | `inputs/imdb_benchmark.csv`, built by `scripts/prepare_imdb_benchmark.py` | **[Verified]** |

Reproduction targets for phase 5, from the parent project's retained outputs.
Each target's status in the agent's own tables is stated; only reproduced
targets are shown as such.

| Target (parent project) | Agent table | Status |
|---|---|---|
| AI-vs-manual κ = 0.9399 | `results/t03_kappa.csv` (0.939910) | reproduced |
| AI-vs-IMDb κ = 0.8899 | `results/t13_reference_agreement.csv` (0.889890) | reproduced |
| manual-vs-IMDb κ = 0.850 | `results/t13_reference_agreement.csv` (0.849910) | reproduced |
| confusion 95/90/8/7 (manual vs IMDb) | — | not tabulated (only its kappa is) |
| McNemar p = 0.21875 | — | not produced (§4: not implemented) |

The three kappa targets are asserted by
`tests/test_reproduce_case_studies.py`, which recomputes them from the frozen
predictions and the sealed benchmark through an independent code path.

Known limitations carried over: star-derived labels are not truth; no
human–human agreement estimate exists (parent U1/U2); pretraining contamination
cannot be excluded (parent U5).

---

## 6. Case study 2 — Financial PhraseBank (3 classes, ordered)

**[Verified]** from the dataset card and the loader script on 2026-11-16:

| Item | Value |
|---|---|
| Source | Hugging Face `takala/financial_phrasebank`, `data/FinancialPhraseBank-v1.0.zip` |
| Accessibility | Public, no token. `HTTP 302 → 200`, `content-type: application/zip`, `content-length: 681890`, CDN key `X-Xet-Cas-Uid=public` |
| Licence | **CC BY-NC-SA 3.0** — non-commercial, share-alike |
| Citation | Malo, Sinha, Korhonen, Wallenius, Takala (2014), *JASIST* 65 |
| Annotators | 16 (3 researchers, 13 Aalto University School of Business M.Sc. students); 5–8 annotations per sentence; majority-vote gold |
| Classes | `negative`, `neutral`, `positive` |
| Subsets | `sentences_50agree` 4,846 · `sentences_66agree` 4,217 · `sentences_75agree` **3,453** · `sentences_allagree` 2,264 |
| Splits | none — a single `train` split |
| Encoding / format | ISO-8859-1, one `sentence@label` per line, split on the **last** `@` |
| Loader | script-based, so `datasets` ≥ 2.14 needs `trust_remote_code=True` |

**Correction recorded:** an earlier draft of this plan stated the 75%-agreement
subset as 2,264 rows. That figure is `sentences_allagree`; the 75% subset is
3,453. The config and fetch script use 3,453.

**Documentation inconsistency [External]:** the card's prose says "4840
sentences" while `sentences_50agree` is 4,846. Not resolved by guessing; noted
for the report.

**[Decision]** Fetch the archive directly and parse its text files instead of
`load_dataset(..., trust_remote_code=True)`: no remote code execution, no
`datasets` dependency, and the parse rule is visible in our own source.

**[Decision]** 300 rows, seeded, **unstratified** draw from the 75%-agreement
subset. Stratifying would fix the class distribution and inflate kappa, which
depends on the marginals.

**Human–human agreement limitation:** the release ships majority labels only, so
per-annotator agreement cannot be recomputed. The published figure is
item-level agreement, not pairwise kappa, and the two are not directly
comparable. The report will say this rather than place them side by side.

---

## 7. Analysis choices made in phase 0–1

1. Paths resolve relative to the config file's directory, so a study folder is
   movable and examples run from any working directory. **[Decision]**
2. Unknown config keys are rejected rather than ignored, so a typo cannot
   silently disable a check. **[Decision]**
3. Classes are a slug (`^[A-Za-z0-9][A-Za-z0-9_.-]*$`) so they can appear
   verbatim in an `ANSWER: <class>` contract without quoting. **[Decision]**
4. `labels.ordered=true` requires `evaluation.kappa` to include `quadratic`, and
   vice versa; the inconsistent combination is a config error rather than a
   silent omission. **[Decision]**
5. ID normalisation trims whitespace and renders integral floats without `.0`,
   so `1`, `"1"`, and `1.0` join. Null IDs stay null and fail a check. **[Decision]**
6. A missing API key is recorded, not fatal, at configure time; it becomes fatal
   before any classification call. **[Decision]**
7. Ambiguity produces *questions*, not interactive prompts: the audit exits with
   code 2 and names the config field that resolves each question, so a re-run is
   non-interactive. **[Decision]**
8. Every filter prints rows before → after with a reason, and the same records
   are stored in `manifest.json → row_accounting`. **[Decision]**
9. Reused predictions are validated like any other input (IDs, domain, status,
   single model) and never edited. **[Decision]**
10. The audit does not write a combined or modified dataset; it only reads. The
    benchmark file is prepared by a separate, visibly non-core adapter script.
11. **Sampling (phase 2):** seeded, unstratified, `unique_text` applied only
    when a subset is drawn. The order is shuffled so the classifier's input does
    not follow the source file, and the sealed file follows the blind order.
    **[Decision]**
12. **Pilot gate (phase 3):** `pilot.n = 5` observations, then an approval file
    that binds the model name, the prompt-template hash, and the pilot
    predictions hash. A changed class definition or a replaced pilot file
    invalidates the approval. **[Decision]**
13. **Temperature is null for both case studies.** It is not controlled through
    the current CLI configuration and is not injected through provider
    request-body config. Every run records `not_requested`. **[Decision]**
14. **Budget is counted from the attempt logs**, pilot and full together, not
    from an in-memory counter and not per file. A resume cannot reset or bypass
    the cap, and the configured number is the true total for the run. Cost is
    taken from the provider's own per-call `step_finish` figures and summed from
    the logs, so a no-op resume cannot zero it; the character-based estimate is
    kept as a labelled floor. **[Decision]**
15. **Predictions that cannot be parsed are never imputed.** They keep an empty
    label and a status; the evaluation stage will report them under both
    abstention policies. **[Decision]**
16. **The pilot report may compare against an earlier prediction file** (the
    IMDb case study reuses one), because predictions are not benchmark labels.
    It never compares against the benchmark. **[Decision]**

---

## 8. Not yet implemented (explicit, so nothing looks finished that is not)

| Phase | Component | Status |
|---|---|---|
| 2 | blind input + benchmark seal + seeded sampling | **built** |
| 3 | provider adapters, prompt builder, parser, resumable classifier, pilot gate | **built** |
| 4 | evaluation metrics, CIs, freeze/join, disagreement tables | **built and exercised** |
| 5 | disagreement analysis, text features, subgroups | **built** (one exploratory length test; pre-specified subgroup contrasts and a three-way reference decomposition in stage 06) |
| 6 | report generator, figures, cross-dataset comparison | **built** |
| 7 | Streamlit interface (`app/`), study report, policy reconciliation | **built** |
| 8 | qualitative disagreement analysis (stage 08), control sample, narrative report | **built** |
| 9 | per-study and comparison report composers, PDF export (stage 09) | **built** |
| 10 | packaging (zip), clean-environment smoke test | not built |

`scripts/fetch_dataset.py` exists and downloads the public archive; it is the
only networking script in the folder. Both adapters are implemented. The
`opencode_cli` adapter has been exercised against a live endpoint: the IMDb
five-row pilot and qualitative stages, the PhraseBank pilot, full classification,
and qualitative stages, and the real three-step interface acceptance run all made
live calls. The `openai_compatible` adapter is implemented and covered by tests
but has not been called against a live endpoint. Every **test** still uses a stub
provider, so the automated suite makes no model call.

Both case studies are past the pilot gate and evaluated: `runs/imdb_case_study`
(200 predictions imported verbatim, plus live pilot and qualitative calls) and
`runs/phrasebank_75agree` (300 live predictions), with the cross-dataset material
in `comparison/`. Each run's `blind/` and `sealed/` hold the blind input and the
sealed benchmark; provenance is recorded in `manifest.json`,
`results/run_metadata.json`, and `VERIFICATION.md`.

---

## Changelog

- **v0.11 (2026-11-20)** — **documentation consistency: current status,
  interface, and limitations. Documentation only.** No method, statistic,
  prediction file, result table, or generated report was changed. §8 no longer
  implies the adapters were never used live or that the case studies stopped at
  the pilot: `opencode_cli` has been exercised live (IMDb five-row pilot and
  qualitative; PhraseBank pilot, full classification, and qualitative; and the
  real three-step acceptance run), `openai_compatible` is implemented but not
  called live, and both case studies are evaluated with `comparison/` generated.
  §3 adds a clarification that the Simple-mode proposal reads the benchmark's
  label column only to propose category names and a mapping for human
  confirmation, so no row-level benchmark label, count, or distribution enters a
  prompt. §4 clarifies that the difficulty flags are multi-label and overlap, so
  per-subgroup n values must **not** be required to sum to the analysed n (a
  deliberate non-check, not an omission); the implemented-versus-unimplemented
  statistical distinction is unchanged. The README, `acceptance/GUIDE.md`, and
  the README's limitations section were updated in step with this entry.

- **v0.10 (2026-11-20)** — **documentation correction: report wording, and the
  version label.** The IMDb case-study report and the cross-dataset comparison
  report were regenerated with the precise reference names introduced in v0.8:
  "manual human annotation" versus "star-derived IMDb proxy label", with an
  explicit statement that a secondary reference may be a proxy label rather than
  a second human annotation. The v0.8 changelog note that "the frozen reports
  retain the earlier wording" therefore no longer applies. This is a
  documentation-only change: no statistical method, prediction file, result
  table, or qualitative explanation was changed or regenerated, and the reports
  remain composed only from the existing result tables. The document title's
  version label is corrected from the stale `v0.7` to `v0.10` to match this
  changelog.

- **v0.9 (2026-11-19)** — **verification pass.** No statistical method was added
  or changed. §4 now separates the implemented evaluation outputs (`t01`–`t16`)
  from documented but unimplemented checks — McNemar, Holm-adjusted p,
  minimum detectable difference, dual abstention-policy tables, and subgroup
  n-sum reconciliation — and §5 marks each parent-project reproduction target as
  reproduced or not tabulated. Added committed verification tests
  (`tests/test_reproduce_case_studies.py`,
  `tests/test_manifest_accounting.py`,
  `tests/test_acceptance_simple_interface_run.py`), a `config_history.json`
  record of every run's configuration hashes, and `VERIFICATION.md`. No
  prediction file, result table, or qualitative explanation was changed or
  regenerated.

- **v0.8 (2026-11-19)** — **subgroup hypothesis tests, the three-way reference
  decomposition, a Reflection document, and documentation corrections.** Stage 06
  now also writes `t14_subgroup_tests.csv` (pre-specified subgroup contrasts with
  Newcombe risk-difference intervals, two-sided Fisher exact tests, and
  Benjamini–Hochberg q-values across the pre-specified flags only),
  `t15_three_way_agreement.csv` (AI prediction vs primary and secondary
  reference), and `t16_reference_dispute.csv` (reference disagreement by
  subgroup) when the configuration declares `evaluation.subgroup_columns` or an
  extra reference. The per-study and comparison reports render these when
  present, with an explicit hypothesis verdict. `configs/imdb_case_study.yaml`
  declares the reference subgroup, two composites, the plan hypothesis, the
  hypothesis reference, and the primary-reference rationale. The word "human"
  was removed from the three-way wording because the IMDb secondary reference is
  a star-derived proxy label, not a second human annotator. Documentation
  corrections: the README status line, the stage-06 output list, the
  evaluation-outputs table, the IMDb case-study call description, and this
  section's phase-5 status. No prediction file, frozen result table, or shipped
  case-study report was changed; the frozen reports retain the earlier wording.
  `REFLECTION.md` was added.

- **v0.7 (2026-11-17)** — phase 9: **report composers and PDF export**. A new
  evaluation-side package `src/general_agent/report/` composes one unique
  per-study report from the existing result tables (`t01`–`t13`) and, when stage
  08 has run, the qualitative tables; otherwise it states explicitly that the
  qualitative analysis has not been run and includes no explanations. The
  cross-dataset report from stage 07 is preserved as a separate document, and
  the qualitative narrative remains its own document. All three are exported to
  PDF with ReportLab (optional `pdf` extra; figures are re-rendered as PNG from
  the same CSVs, so no new numbers and no new calculation). The Streamlit
  Results and Case Studies pages gain report buttons. The section builders in
  `evaluate/compare.py` and `qualitative/narrative.py` were extracted for reuse;
  their existing Markdown outputs are unchanged, and the statistical engine,
  frozen predictions, `results/`, `comparison/`, and the two case-study run
  directories are untouched. Composed reports are written to a new gitignored
  `reports/` directory.

- **v0.6 (2026-11-17)** — phase 8: a reusable, evaluation-side **qualitative
  disagreement analysis** (`src/general_agent/qualitative/`, CLI
  `scripts/08_qualitative.py`). It reads the frozen predictions and the sealed
  reference (only after stage 05 has frozen), samples a seeded set of
  disagreements and a small seeded **control** sample of agreements, and asks a
  model for a structured explanation of each. Explanations are labelled
  hypotheses, never confirmed causes or proven errors. Recurring reasons are
  counted deterministically from a versioned controlled vocabulary (no
  clustering call), and a template-composed narrative embeds the stage 07
  statistical report. Settings live in a sidecar
  (`runs/<run_id>/qualitative/settings.yaml`) so the sealed `study.yaml` is
  never edited; the qualitative budget and attempt log are separate from
  classification, and a three-item pilot must be approved before any further
  calls. The Streamlit Results page gains a gated qualitative section with the
  narrative and tables. The statistical engine, `results/`, `predictions.csv`,
  and the two case studies are unchanged. No number in the narrative is written
  by the model.

- **v0.5 (2026-11-17)** — phase 7: Streamlit interface (`app/`) that drives the
  existing stages without reimplementing any evaluation logic; the generated
  per-study report is the existing stage 07 generator run with one config, and
  its heading is now "Study report" rather than "Cross-dataset comparison".
  Three method-neutral corrections found while building the interface: stage 02
  now materialises the ID intersection and primary-label exclusion policies that
  the audit validated in memory (previously the blind input and sealed benchmark
  could disagree whenever `on_id_mismatch: inner_join` or
  `on_unknown_label: exclude` was used); the stage 05 evaluation log now reports
  the real failed-prediction count from `t01` instead of a hardcoded zero, and
  its reference sentence comes from the config rather than the PhraseBank study;
  and the report heading reflects how many studies it covers. `streamlit` is an
  optional `ui` extra; the core runtime dependencies are unchanged. No metric,
  interval, sampling, or blinding rule changed; `runs/imdb_case_study`,
  `runs/phrasebank_75agree`, and `comparison/` are untouched.

- **v0.4 (2026-11-16)** — phase 6: reused-prediction import, multi-reference
  join, disagreement pair/text tables, secondary-reference agreement,
  cross-dataset comparison, generated report and figures, and `MISTAKES.md`.
  Adds stages 06–07 and a genericity test. Evaluation now reads only the sealed
  reference and applies `label_map` to every reference.

- **v0.3 (2026-11-16)** — phase 5: freeze-before-benchmark, sealed-only
  reference read, join reconciliation, accuracy/kappa/per-class metrics,
  Wilson and seeded-bootstrap CIs, ordinal error structure, disagreement
  table, and an explicit confirmed-vs-unconfirmed error table. Adds
  `evaluate/` and `scripts/05_evaluate.py`. Two defects were found by the real
  data and fixed before the reported numbers: the evaluation initially read the
  full 3,453-row source instead of the sealed 300, and the benchmark's `label`
  column collided with the predictions' `label` column during the merge.

- **v0.2 (2026-11-16)** — phases 2–3: seeded unstratified sampling, blind input,
  sealed benchmark with a directory lock, both provider adapters, prompt builder
  with a render guard, strict answer parser, pilot gate bound to the model and
  prompt hash, resumable atomic classifier, budget counting from the attempt
  log. Decisions 11–16 added. No live model calls; all tests use a stub
  provider. Audit now redacts benchmark example values from `decisions.json`.
- **v0.1 (2026-11-16)** — phase 0–1: specification, config schema, config
  loader, column discovery with clarification policy, dataset audit, tests.
  Temperature capability verified (config-body injection only); Financial
  PhraseBank accessibility, licence, and agreement counts verified; 75%-subset
  count corrected from 2,264 to 3,453.
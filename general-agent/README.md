# general-agent

A general-purpose agent that classifies observations with an AI model and
compares the predictions against a **human benchmark** — any single-label task
with two or more classes, not just IMDb sentiment.

Classification and evaluation are kept apart on purpose: the model never sees a
benchmark label, the benchmark is sealed before any call, and evaluation only
runs after predictions are frozen.

> **Status: phases 0–9 implemented** — the engine, the Streamlit interface, the
> qualitative disagreement analysis, and the per-study / comparison report
> composers. Configure, audit, blind/seal, pilot, classification, evaluation,
> disagreement analysis, subgroup contrasts, qualitative explanations, and
> generated reports work for both studies. The `app/` interface drives the same
> engine for any dataset. Packaging is not built yet; `SPEC.md` §8 lists exactly
> what is missing.

---

## Install

```bash
cd general-agent
uv sync --extra dev          # creates .venv and installs dependencies
```

Without `uv`:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Python 3.11 or newer. Runtime dependencies: `pandas`, `pyyaml`, `jsonschema`.
The HTTP provider uses only the standard library.

For the Streamlit interface, add the optional extra:

```bash
uv sync --extra ui --extra dev
uv run streamlit run app/streamlit_app.py
```

The interface binds to `127.0.0.1`, never asks for a credential, and reuses the
stages below rather than reimplementing them. Its default three-step flow is the
**recommended workflow** for a new study (see *Recommended workflow* below); the
original seven-step flow is kept behind **Advanced mode**. See `app/README.md`.

For PDF report export, add the ReportLab extra:

```bash
uv sync --extra ui --extra pdf --extra dev
```

Without it, every report is still produced as Markdown.

---

## Recommended workflow: the three-step interface

For a new study, the recommended path is the Streamlit interface's default
**three-step flow** (`app/streamlit_app.py`):

1. **Upload** — the observations and a separate human-labelled benchmark.
2. **Review** — the interface *proposes* the ID / text / label columns, category
   names and definitions, the label mapping, the model, and safe call and spend
   caps. Only genuine ambiguities are raised; everything else is collapsed under
   **Advanced settings**. Label polarity is never inferred.
3. **Analyze & Download** — one checkbox authorizes the whole bounded run
   (validate → seal → five-row pilot → blind classification → evaluation →
   disagreements → qualitative explanations only when disagreements exist →
   report and PDF). Progress, calls used, provider cost, **Stop**, **Resume**, and
   the downloads are on this screen.

The five-row pilot is **approved automatically** once every row parses to a
declared category; if a pilot row fails, the run pauses and asks. One
authorization covers the run; changing the model or raising a cap requires a new
one.

The original **seven-step flow** stays available behind the **Advanced mode**
button for anyone who wants to control each stage: Data → Columns → Categories →
Model and limits → Validate and seal → Classify (run the pilot, then **approve it
manually**) → Results. Both flows drive the same engine with the same safeguards
— the same seal, pilot gate, freeze-before-read, and caps.

---

## The command line (stages 00–09)

```bash
# 0. check the config, resolve paths, list anything missing
uv run python scripts/00_configure.py --config configs/case2_phrasebank.yaml

# 1. audit the data: IDs, labels, leakage, row counts
uv run python scripts/01_audit.py --config configs/case2_phrasebank.yaml

# 2. build the blind input and seal the benchmark
uv run python scripts/02_blind.py --config configs/case2_phrasebank.yaml

# 3. classify pilot.n observations, review, then approve
uv run python scripts/03_pilot.py --config configs/case2_phrasebank.yaml
uv run python scripts/03_pilot.py --config configs/case2_phrasebank.yaml --approve

# 4. full run (requires the approval file)
uv run python scripts/04_classify.py --config configs/case2_phrasebank.yaml

# 5. freeze predictions, then open the benchmark and compute metrics
uv run python scripts/05_evaluate.py --config configs/case2_phrasebank.yaml

# 6. descriptive disagreement patterns
uv run python scripts/06_disagreements.py --config configs/case2_phrasebank.yaml

# 7. cross-dataset tables, figures, and report
uv run python scripts/07_compare.py --configs \
    configs/imdb_case_study.yaml configs/case2_phrasebank.yaml

# 8. qualitative explanation of disagreements (AI hypotheses; opt-in)
uv run python scripts/08_qualitative.py --config configs/case2_phrasebank.yaml \
    --settings runs/phrasebank_75agree/qualitative/settings.yaml --pilot
# review, then --approve, --run, and --report (see app/README.md)

# 9. compose the per-study report and export PDFs (no model call)
uv run python scripts/09_report.py --config configs/case2_phrasebank.yaml
uv run python scripts/09_report.py --all          # every evaluated run
uv run python scripts/09_report.py --comparison   # the cross-dataset report
```

Exit codes: **0** success, **1** failure or incomplete, **2** blocked by a
precondition (clarification, missing approval), **3** stopped early (budget or a
fatal provider error).

---

## What each stage writes

Everything lives under `runs/<run_id>/` (gitignored).

| Stage | Output |
|---|---|
| 00 configure | `manifest.json` starts here: config hash, resolved paths, defaults *actually* applied |
| 01 audit | `decisions.json` (mapping + every automatic choice), `audit.txt`, manifest checks |
| 02 blind | `blind/blind.csv` (only `id`, `text`), `sealed/benchmark.csv`, sample record in the manifest |
| 03 pilot | `pilot/pilot_predictions.csv`, `pilot/pilot_attempts.jsonl`, `pilot/pilot_report.json`, then `pilot/approval.json` |
| 04 classify | `predictions.csv`, `logs/attempts.jsonl`, manifest counts |
| 05 evaluate | `results/`: frozen copy, 10 metric tables, `evaluation_log.txt`, `run_metadata.json` |
| 06 disagreements | `results/t11`–`t16` (t14–t16 when subgroups or a secondary reference are declared), `disagreement_log.txt` |
| 07 compare | `comparison/`: 6 tables, 3 SVG figures, `report.md` |
| 08 qualitative | `qualitative/`: selection, explanations, pattern tables, `narrative_report.md`, attempt log, approval |
| 09 report | `reports/<run_id>/study_report.md` (+ `.pdf`), `reports/<run_id>/qualitative_report.pdf`, `reports/comparison/report.pdf` |

No configured input is ever modified. The composed reports live in `reports/`
(gitignored) so the shipped case-study run directories and `comparison/` are left
byte-identical.

---

## Using your own data

You need two files.

**Dataset to classify** — an ID column and a text column; extra metadata is
allowed but never sent to the model.

```csv
id,text
1,"The battery lasts two days, but the screen scratches easily."
2,"Arrived broken and support ignored me for a week."
```

**Human benchmark** — matching IDs and the human label.

```csv
id,label
1,mixed
2,negative
```

Copy `configs/template.yaml`, set the paths, declare the classes, and run the
first two commands. Paths resolve relative to the config file.

The audit resolves unambiguous columns and **asks** (exit 2) when a wrong guess
would corrupt the evaluation: two plausible text columns, numeric labels without
a `label_map`, mismatched IDs, duplicate IDs, or labels outside the declared
classes. Answer by editing the named config field; re-runs are non-interactive.

### Sampling

```yaml
sample:
  n: 300               # null or absent = every row
  unique_text: true
  method: random       # seeded shuffle, NEVER stratified
```

The draw uses `run.seed` and is deterministic. It is deliberately not
stratified: fixing the class distribution would inflate kappa, which depends on
the marginals. `unique_text` skips a drawn row whose text was already selected,
and only applies when a subset is drawn.

---

## The pilot gate

The three-step interface approves a cleanly parsed pilot automatically; the
manual approval below is the Advanced-mode and command-line path. Stage 03
classifies `pilot.n` observations and stops. Review
`pilot/pilot_predictions.csv`, then approve:

```bash
uv run python scripts/03_pilot.py --config ... --approve --note "5/5 parseable"
```

The approval records the model name, the prompt-template hash, and the pilot
file hash. Stage 04 refuses to run if the model or any part of the prompt
template has changed since approval — a changed class definition invalidates the
pilot. If the config reuses an earlier prediction file, the pilot report also
shows agreement against it (predictions only; the benchmark stays sealed).

---

## Budget, resume, and provider failures

```yaml
budget:
  max_calls: 320       # total for the run_id, across pilot and full logs
  max_usd: 0.50        # guardrail; provider-reported cost is what gets recorded
```

- Every attempt is one call. `max_calls` counts **both** the pilot log and the
  full-run log, so 5 pilot calls + 300 full calls needs 305, not 300 plus a free
  pilot. The runner stops before exceeding the cap and leaves untouched rows
  `pending`.
- The OpenCode CLI reports per-call cost and token usage in its `step_finish`
  output. `actual_usd` is summed from the attempt logs (surviving resumes); the
  character-based estimate counts only your prompt and is a **floor**, because
  it excludes the harness's cached context and reasoning tokens.
- `--resume` continues a file, skipping rows whose status is `ok`.
- `--limit N` processes at most N pending rows in one invocation.
- A fatal provider error (`blocked_auth`, `blocked_billing`,
  `blocked_rate_limit`) stops immediately: retrying cannot succeed and burns
  budget.
- Missing predictions are never imputed. They keep an empty label and a status.
- Each attempt is appended to a JSONL log with the full prompt, the raw reply
  (flagged if truncated), model, attempt number, timing, and outcome.

---

## The seal

Stage 02 writes two files and never places them together:

```
runs/<run_id>/blind/blind.csv        id, text only
runs/<run_id>/sealed/benchmark.csv   ids, reference labels, subgroup metadata
```

During stage 04 the sealed directory is `chmod 000`, then restored in a
`finally` block. A same-user process cannot traverse it while the model runs.
This is a barrier, not a sandbox (`SPEC.md` §3 lists each layer and its limits).

If a run is killed hard enough to skip the restore:

```bash
uv run python scripts/02_blind.py --config ... --unlock
```

---

## Comparison outputs

`comparison/` holds the cross-dataset material, generated from result tables
only:

| File | Contents |
|---|---|
| `tables/cross_dataset_metrics.csv` | accuracy, balanced accuracy, kappa with CIs |
| `tables/cross_dataset_classes.csv` | per-class precision/recall/F1 by dataset |
| `tables/cross_dataset_severity.csv` | agree/adjacent/opposite shares |
| `tables/cross_dataset_references.csv` | secondary-reference agreement, when one exists |
| `tables/cross_dataset_pairs.csv` | disagreement directions per dataset |
| `tables/cross_dataset_text.csv` | median observation length by agreement group, exploratory test |
| `figures/*.svg` | accuracy/kappa, per-class recall, ordinal severity |
| `report.md` | generated cross-dataset report |
| `run_metadata.json` | hashes of every source table |

The comparison code contains no dataset names; study-specific text comes from
each config's `report` block.

---

## Evaluation outputs

`results/` holds, for the analysed rows:

| File | Contents |
|---|---|
| `t01_join_and_sample.csv` | every row count from freeze to analysed set, with reasons |
| `t02_accuracy.csv` | n, correct, accuracy (Wilson CI), balanced accuracy (bootstrap CI) |
| `t03_kappa.csv` | unweighted and quadratic-weighted kappa with bootstrap CIs |
| `t04_confusion_matrix.csv` | rows = human, columns = AI, plus marginals |
| `t05_per_class_metrics.csv` | per-class precision/recall/F1 with CIs and support |
| `t06_class_distribution.csv` | human vs AI marginals and the bias |
| `t07_ordinal_errors.csv` | agree / adjacent / opposite counts with Wilson CIs |
| `t08_disagreements.csv` | one row per disagreement, labelled **unconfirmed** |
| `t09_category_summary.csv` | per-reference-class recall and error targets |
| `t10_error_status.csv` | confirmed errors (0 here) vs unconfirmed disagreements |
| `t11_disagreement_pairs.csv` | reference → prediction counts for every pair |
| `t12_text_profile.csv`, `t12b_length_test.csv` | observation-length profile and seeded permutation test |
| `t13_reference_agreement.csv` | agreement with any declared secondary reference |
| `t14_subgroup_tests.csv` | pre-specified subgroup contrasts (risk difference, Fisher p, BH q) |
| `t15_three_way_agreement.csv` | AI vs primary vs secondary reference decomposition |
| `t16_reference_dispute.csv` | reference-disagreement rate by subgroup |

A disagreement is not a model error: the reference is a human or proxy
benchmark, not adjudicated truth, and where per-annotator labels are not
released human–human agreement cannot be computed. For IMDb the primary
reference is a single manual annotation and the secondary is a star-derived
proxy label; for PhraseBank the reference is a panel majority. The tables say so
rather than blurring the distinction.

---

## Model access and credentials

| Provider | Credentials | Temperature |
|---|---|---|
| `opencode_cli` | none; uses the local binary | **not supported** — the CLI has no temperature flag |
| `openai_compatible` | key read from the env var named in `model.api_key_env` | sent in the request body when set |

A config never contains a key, and `.env` is gitignored.

**Both case studies set `temperature: null`.** Even if a value were set, the
manifest records it as `requested_unverified` or `not_supported`; the report may
never claim a controlled temperature. If you need determinism evidence, use the
pilot/stability comparison, not the config value.

The OpenCode provider also has no per-call temperature control because the value
would have to be injected through provider request-body configuration, which
this project does not do. That is a documented limitation, not an oversight.

---

## The two shipped case studies

**IMDb (binary; the 200 classification predictions are reused, not re-run).**

```bash
uv run python scripts/prepare_imdb_benchmark.py
uv run python scripts/00_configure.py --config configs/imdb_case_study.yaml
uv run python scripts/01_audit.py    --config configs/imdb_case_study.yaml
uv run python scripts/02_blind.py    --config configs/imdb_case_study.yaml
```

200 rows; primary reference = the 200 manual annotations (a human reading of the
text); secondary reference = the star-derived dataset label (a proxy derived from
ratings, not a second human annotator). The 200 classification predictions were
imported verbatim — `runs/imdb_case_study/predictions_import.json` records zero
classification calls. The five-row stage-03 pilot and the qualitative
explanation stage did make live calls.
[Corrected 2026-11-19: an earlier version said the pilot had not been run and did
not mention the qualitative calls.]

**Financial PhraseBank (3 classes, ordinal, panel-annotated).**

```bash
uv run python scripts/fetch_dataset.py --agreement 75
uv run python scripts/00_configure.py --config configs/case2_phrasebank.yaml
uv run python scripts/01_audit.py    --config configs/case2_phrasebank.yaml
uv run python scripts/02_blind.py    --config configs/case2_phrasebank.yaml
```

3,453 sentences available; 300 sampled for analysis. Licence **CC BY-NC-SA 3.0**
(non-commercial, share-alike): data is fetched into gitignored `inputs/` and is
not redistributed with this agent.

---

## Methodological limitations and non-delivered checks

These are stated once so that nothing below reads as more than it is.

- **Reference labels are benchmarks, not adjudicated truth.** Every disagreement
  is reported as **unconfirmed**, and the agent reports zero confirmed model
  errors (`t10_error_status.csv`).
- **The IMDb star-derived label is a proxy, not a second human annotator.** The
  primary reference is one person's text reading; the secondary reference is
  derived from star ratings (`../data/README.md` §5). Neither is truth.
- **No reliability estimate exists.** The annotation codebook planned an
  intra-annotator re-annotation of ~20 reviews (`../codebook.md` §7); it was not
  completed, and the annotation has a single annotator, so neither intra- nor
  inter-annotator agreement can be reported (`../mistakes.md` U1, U2). This is a
  planned check that was **not carried out**, not a reported null result.
- **Subgroup flags overlap.** `sarcasm`, `mixed`, `ambiguous`, and `complex` are
  independent multi-label flags (`../codebook.md` §4), so one review can carry
  several. Their per-flag sample sizes therefore **must not** be required to sum
  to the analysed total, and the agent does not assert such a sum.
- **Some checks are documented but not delivered** (`SPEC.md` §4): the McNemar
  exact test, Holm-adjusted p-values, dual abstention-policy tables, and a
  minimum detectable difference. No MDE is computed for this agent, so "not
  significant" is never reported as "no difference". The 19–29 pp MDE in
  `../mistakes.md` U3 belongs to the earlier group project's power analysis, not
  to this agent's output.

---

## Keeping results for the final submission

`runs/<run_id>/` (per-run artefacts) and `reports/` (composed per-study and
comparison reports) are **gitignored**, so a commit or clone does not carry them.
The cross-dataset material under `comparison/` is not gitignored. The required
case-study results — the evaluated `runs/imdb_case_study` and
`runs/phrasebank_75agree` directories and the composed reports you rely on — must
therefore be preserved explicitly when the final submission is prepared. This
project does not create a ZIP and does not assume a particular submission format.

---

## Reproducibility

- `run.seed` drives sampling and every later random operation; it is recorded in
  each manifest.
- Every stage records the hash of its input and output.
- `manifest.json → stages` accumulates one section per stage, including the
  provider description, prompt-template hash, call count, and stop reason.
- `decisions.json` records every automatic column choice and its reason.
- The approval file binds the pilot to a specific model and prompt template.

A reader can therefore check, without trusting the prose, which data, model,
prompt, and sample produced a prediction.

---

## Tests

```bash
uv run pytest
```

The suite covers config validation, clarification policy, sampling determinism,
blinding, leakage (poison marker, import boundary, file-access scope), provider
transport parsing, fatal-error classification, the approval gate, resume,
budget caps, and atomic saves. No test performs a model call.

---

## Folder map

```
app/            Streamlit interface (optional; see app/README.md)
configs/        schema.json, template.yaml, ui_providers.yaml, qualitative_*.{json,yaml}, one YAML per study
src/general_agent/
  config.py     validation, defaults, path resolution
  io.py         CSV/Parquet/JSONL reading, ID normalisation, hashing
  discover.py   column discovery and the clarification policy
  audit.py      stage 01
  blind.py      stage 02: sampling, blind input, sealed benchmark
  sealing.py    directory lock used during classification
  prompts.py    prompt construction and hash guard
  parsing.py    ANSWER-line parsing
  classify.py   stages 03-04: pilot, approval, full run
  evaluate/     stage 05: freeze, join, metrics, intervals, tables
                stages 06-07: disagreements, comparison, generated report
  qualitative/  stage 08: explanation pilot, approval, patterns, narrative
  report/       stage 09: per-study report, Markdown->PDF, figures
  providers/    opencode_cli and openai_compatible adapters
  cli.py        command surface
scripts/        00_configure … 09_report, plus case-study adapters
tests/          fixtures, stubs, and tests
runs/           per-run outputs (gitignored)
reports/        composed per-study and comparison reports (gitignored)
inputs/         fetched or prepared data (gitignored)
SPEC.md         frozen method record and open items
AGENTS.md       project rules
```
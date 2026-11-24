# Individual AI Project — Can an AI Classifier Be Trusted?

This repository is the final submission for Professor Pandey's Individual AI Project. It contains a reusable agent that classifies text with an AI model and compares those predictions against a separate human benchmark, together with two case studies: IMDb sentiment (binary) and Financial PhraseBank sentiment (three classes).

The project is not a single hard-coded analysis. It asks a practical question: when an AI classifier disagrees with a human reference, what can we actually conclude? The answer depends on a protocol that keeps prediction and evaluation apart, and on being honest about what the evidence can and cannot establish.

**Submission documents**

| Document | File |
|---|---|
| Reflection | [`general-agent/REFLECTION_SUBMISSION.md`](general-agent/REFLECTION_SUBMISSION.md) · [PDF](general-agent/REFLECTION_SUBMISSION.pdf) |
| Error Analysis | [`general-agent/ERROR_ANALYSIS_SUBMISSION.md`](general-agent/ERROR_ANALYSIS_SUBMISSION.md) · [PDF](general-agent/ERROR_ANALYSIS_SUBMISSION.pdf) |
| Technical guide | [`general-agent/README.md`](general-agent/README.md) |
| Method record | [`general-agent/SPEC.md`](general-agent/SPEC.md) |
| Verification record | [`general-agent/VERIFICATION.md`](general-agent/VERIFICATION.md) |
| Error register | [`general-agent/MISTAKES.md`](general-agent/MISTAKES.md) |

## 1. Purpose and research question

The project asks how accurately an AI model can classify text, and which cases produce disagreement with a human benchmark. It also asks a second, methodological question: how should an AI-assisted analysis be checked?

The agent therefore enforces a separation that is easy to state and harder to maintain:

- the classifier reads only an identifier and the observation text;
- the human labels live in a sealed benchmark;
- predictions are frozen and hashed before the benchmark is opened;
- a small pilot is bound to the model and prompt before a full run;
- call and spend caps are counted from attempt logs.

These are internal controls. They make the pipeline auditable; they do not make it independently verified. No causal claims are made, and a disagreement is never treated as a confirmed model error.

## 2. What the agent does

The Streamlit interface offers a default **three-step flow**:

1. **Upload** — the observations to classify and a separate human-labelled benchmark.
2. **Review** — the interface proposes the ID, text and label columns, category names, mapping, model and caps. Genuine ambiguities are raised; label polarity is never inferred.
3. **Analyze & Download** — one authorization runs the bounded pipeline: audit → seal → five-row pilot → blind classification → evaluation → disagreements → optional qualitative explanations → report.

One authorization covers the run. Changing the model or raising a cap requires a new authorization. A cleanly parsed five-row pilot is approved automatically in Simple mode; the original seven-step flow remains available under **Advanced mode**, where the pilot is approved manually.

| Stage | What it does | Main evidence |
|---|---|---|
| Audit | Checks IDs, labels, leakage and row counts | `decisions.json`, `audit.txt` |
| Blind and seal | Writes `id, text` only; seals the benchmark | `blind/blind.csv`, `sealed/benchmark.csv` |
| Pilot | Five-row check bound to model and prompt | `pilot/` |
| Classify | Live calls with caps and attempt logs | `predictions.csv`, `logs/attempts.jsonl` |
| Evaluate | Freezes predictions, joins after unsealing | `results/t01`–`t16` |
| Report | Composes documents from result tables only | `reports/<run_id>/` |

## 3. Repository structure and where to find each component

```text
README.md                         this landing page
general-agent/                    the individual project
  README.md                       technical guide and commands
  SPEC.md                         frozen method record
  MISTAKES.md                     23-entry individual error register
  VERIFICATION.md                 verification evidence and limits
  REFLECTION.md                   original internal reflection
  REFLECTION_SUBMISSION.md/.pdf   final reflection
  ERROR_ANALYSIS_SUBMISSION.md/.pdf
  app/                            Streamlit interface
  src/general_agent/              engine
  scripts/                        00_configure … 09_report
  tests/                          no-model-call test suite
  configs/                        study configs and provider allow-list
  acceptance/                     synthetic UI acceptance kit
  comparison/                     cross-dataset tables, figures and report
  runs/                           committed case-study result tables, frozen
                                  predictions, sealed benchmarks and predictions;
                                  blind text, logs and full runs are gitignored
  reports/                        original composed reports (gitignored)
  publication/                    sanitized reports, provenance and aggregate tables
data/                             minimal project inputs
codebook.md                       annotation codebook
plan.md                           original project plan
mistakes.md                       earlier group-project error record
results/                          selected earlier group-project context
```

| Need | Where |
|---|---|
| Reflection | [`general-agent/REFLECTION_SUBMISSION.md`](general-agent/REFLECTION_SUBMISSION.md) |
| Error Analysis | [`general-agent/ERROR_ANALYSIS_SUBMISSION.md`](general-agent/ERROR_ANALYSIS_SUBMISSION.md) |
| Code | [`general-agent/src/general_agent/`](general-agent/src/general_agent/) and [`general-agent/scripts/`](general-agent/scripts/) |
| Tests | [`general-agent/tests/`](general-agent/tests/) |
| Case-study result tables | [`general-agent/runs/imdb_case_study/results/`](general-agent/runs/imdb_case_study/results/) and [`general-agent/runs/phrasebank_75agree/results/`](general-agent/runs/phrasebank_75agree/results/) |
| Per-study reports | [`general-agent/publication/reports/imdb_case_study/`](general-agent/publication/reports/imdb_case_study/) and [`general-agent/publication/reports/phrasebank_75agree/`](general-agent/publication/reports/phrasebank_75agree/) |
| Cross-dataset report | [`general-agent/publication/reports/comparison/report.md`](general-agent/publication/reports/comparison/report.md) |
| Verification evidence | [`general-agent/VERIFICATION.md`](general-agent/VERIFICATION.md) and the tests above |

**Packaging note.** `general-agent/reports/` and `general-agent/inputs/` are gitignored, and the blind text, per-call logs and full run directories under `general-agent/runs/` are deliberately not redistributed. The evaluated case-study evidence needed to reproduce the statistics — `results/frozen_predictions.csv`, the `results/t01`–`t16` tables, `sealed/benchmark.csv`, `predictions.csv` and the pilot files for both studies — **is committed and travels with a clone**. A curated, sanitized subset of provenance, context and reports is also committed under [`general-agent/publication/`](general-agent/publication/); the report links above point to that subset. The original run and report directories remain local and are not modified.

## 4. Install and launch

Requirements: Python 3.11 or newer. The project is tested with `uv`; a plain `venv` works too. The agent is self-contained under `general-agent/`.

```bash
git clone <repository-url> my-analysis
cd my-analysis/general-agent
uv sync --extra ui --extra pdf --extra dev
uv run streamlit run app/streamlit_app.py
```

Without `uv`:

```bash
git clone <repository-url> my-analysis
cd my-analysis/general-agent
python -m venv .venv
.venv/bin/pip install -e ".[ui,pdf,dev]"
.venv/bin/streamlit run app/streamlit_app.py
```

The interface binds to `127.0.0.1` only and is intended for single-user local use. It starts with **no provider installed and no model call**: the bundled case studies open read-only, and a model call happens only after the review step and the run authorization. The OpenCode CLI is **not** required to launch the interface or inspect results — only to classify through the `opencode_cli` provider.

## 5. API provider setup

Two providers are selectable in [`general-agent/configs/ui_providers.yaml`](general-agent/configs/ui_providers.yaml):

- **OpenRouter (your own API key)** — `openai_compatible`, enabled, and the **default selection**. Use this to run the agent with your own credentials and no OpenCode account.
- **OpenCode CLI (local)** — `opencode_cli`, enabled. Uses the signed-in local OpenCode CLI; it stores no key in this repository but does require the OpenCode CLI to be installed and signed in. Select it manually in the Provider dropdown to use it.

To classify with your own OpenRouter key, export it in the **same shell** that starts Streamlit:

```bash
export OPENROUTER_API_KEY="sk-or-..."
cd general-agent
uv run streamlit run app/streamlit_app.py
```

OpenRouter is already selected in the interface's Provider dropdown by default, so the key is picked up without changing any setting. The key is read from the named environment variable; it is never accepted, displayed, logged or stored.

- `.env` and `.env.*` are gitignored and are **not loaded automatically** — only variables exported in the shell that launched Streamlit are visible.
- No credentials belong in a config file.
- Temperature is recorded as `not_requested`; it is not controlled by this setup.
- The interface only reports whether the named environment variable is present.
- The OpenRouter model ids and prices in the allow-list were verified against `https://openrouter.ai/api/v1/models` on 2026-11-24; re-check the catalogue before a run, and to point at a different OpenAI-compatible endpoint edit `base_url`, `api_key_env` and `models[].name` in the same file.

## 6. Case studies and key results

| Study | n | Reference | Accuracy | Cohen's κ | Disagreements | Confirmed errors |
|---|---:|---|---:|---:|---:|---:|
| IMDb | 200 | One manual annotator | 0.970 | 0.940 | 6 | 0 |
| Financial PhraseBank | 300 | Majority of 5–8 annotators, ≥75% agreement | 0.913 | 0.841 | 26 | 0 |

Exact values are in `t02_accuracy.csv` and `t03_kappa.csv` under each run's `results/` directory. Accuracy and kappa are descriptive statistics on the analysed samples.

**IMDb.** The 200 predictions were **imported** from the earlier workflow, not regenerated: `predictions_import.json` records zero classification calls. A five-row pilot was run and made five live calls; it parsed 5/5 and matched the reused predictions 5/5. Because stage 04 was not run, no `pilot/approval.json` exists for IMDb. The qualitative stage also made live calls. The primary reference is one person's manual annotation; the secondary reference is a star-derived IMDb proxy label, not a second annotator.

**Financial PhraseBank.** The run contains 300 live predictions from the 3,453-sentence, ≥75%-agreement subset. The five-row pilot was approved and the call cap was raised to 320. The reference is a majority verdict of 5–8 annotators.

**What zero confirmed errors means.** `t10_error_status.csv` reports zero **confirmed** model errors for both studies. This does not mean zero actual errors. Confirmation would require adjudication or per-annotator labels, which are not available. Every disagreement is therefore reported as **unconfirmed**. In IMDb, only 1 of the 6 disagreements is a case where the manual annotation and the star-derived proxy agree and the AI differs; the other 5 are reference disputes. The publication-safe report copies omit illustrative source quotations; all statistical results, figures and limitations are retained.

## 7. Checking existing results without paid model calls

```bash
cd general-agent
uv run pytest
uv run pytest tests/test_reproduce_case_studies.py -rs
uv run python scripts/09_report.py --all
uv run python scripts/09_report.py --comparison
```

- `uv run pytest` uses a stub provider and makes no model call.
- `tests/test_reproduce_case_studies.py` recomputes accuracy and kappa from `runs/*/results/frozen_predictions.csv` and `runs/*/sealed/benchmark.csv`. It **skips** if those files are absent; in this repository that evidence is committed, so the check runs on a fresh clone (`3 passed`).
- `09_report.py` composes reports from existing result tables and makes no model call.
- `05_evaluate.py` and `06_disagreements.py` are deterministic but write into `runs/<run_id>/`; copy a run directory first if you do not want to touch the shipped evidence.
- Avoid `03_pilot.py`, `04_classify.py` and `08_qualitative.py` unless you intend to make paid calls and have approved the pilot and caps.

These commands reproduce the **statistics**. They do not reconstruct the original classification run: that would require the excluded observation text, a configured provider, paid calls and a new pilot authorization. The public package is a statistical verification package, not a self-contained classification rerun.

## 8. Known limitations

- **No independent external verification.** The individual agent's tests and hashes are same-system controls. The independent pass belongs to the earlier group project.
- **No reliability estimate.** The IMDb annotation has one annotator and the planned re-annotation was not completed; the PhraseBank release does not provide per-annotator labels.
- **The references are benchmarks, not truth.** The IMDb secondary label is star-derived; PhraseBank is restricted to the ≥75% subset, which plausibly overstates agreement.
- **Aggregate class counts were seen before classification.** No row-level reference value entered a prompt.
- **Model conditions are only weakly controlled.** Temperature is `not_requested`, endpoint routing is not recorded per call, and there is one model and one prompt.
- **Two convenience samples.** Pretraining contamination cannot be excluded; kappa is sensitive to class prevalence.
- **Some checks are documented but not implemented.** McNemar, Holm-adjusted p-values, minimum detectable difference and dual abstention-policy tables are absent.
- **Qualitative explanations are hypotheses.** The explanation model saw the human label and is the same model that classified; the control sample is small and no multiplicity adjustment applies.
- **The `t08` severity-label inconsistency is unresolved.** For the binary IMDb task, all six rows in `runs/imdb_case_study/results/t08_disagreements.csv` say `severity = adjacent`, while `t07_ordinal_errors.csv` and `comparison/tables/cross_dataset_severity.csv` correctly classify all six as `opposite`. No metric or count changes, and the original CSV has not been edited. Until a versioned fix regenerates it, `t07` and the cross-dataset table are authoritative. See Error Analysis §8.
- **Legacy manifest schema and config-hash drift.** Some old records do not carry split call fields or a classification-time config hash. They are documented in `VERIFICATION.md` and do not indicate a budget violation.

## 9. Individual project versus earlier group-project context

The individual deliverable is `general-agent/`: the agent, its interface, tests, case-study runs, composed reports and the two submission documents.

The earlier group project is context only. Its files are `mistakes.md`, [`general-agent/publication/context/verification_findings_public.md`](general-agent/publication/context/verification_findings_public.md) and [`general-agent/publication/context/agent_autonomous_decisions.md`](general-agent/publication/context/agent_autonomous_decisions.md). They explain why the individual agent was designed as it was, and they supply the independent verification evidence for the group-project numbers. They are not individual-project results. The verification copy has annotator initials redacted; the original local `results/verification_findings.md` is unchanged and is not part of the distributed repository.

In the Error Analysis, individual errors are prefixed `GA-` and group-project errors are prefixed `GP-`. That distinction is deliberate: the two registers reuse local IDs such as M1, M2 and M10.

## 10. Data and licence

Raw datasets are not part of the intended submission package. The Financial PhraseBank is released under CC BY-NC-SA 3.0 (non-commercial, share-alike) and is fetched into the gitignored `general-agent/inputs/` directory at run time. The raw IMDb archive is excluded; the individual case uses the derived 200-row input and the existing prediction file. See [`data/README.md`](data/README.md) and [`general-agent/scripts/fetch_dataset.py`](general-agent/scripts/fetch_dataset.py).
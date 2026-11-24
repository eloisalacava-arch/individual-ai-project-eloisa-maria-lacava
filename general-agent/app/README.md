# Streamlit interface

A thin, local interface over the `general-agent` engine. It builds a normal
study configuration, calls the existing stages, and displays the files they
write. It contains no evaluation logic of its own: every number shown comes from
a CSV under `runs/<run_id>/results/`, and the reports are composed by stages 07
and 09.

## Install and launch

```bash
git clone <repository-url> my-analysis
cd my-analysis/general-agent
uv sync --extra ui --extra pdf --extra dev   # --extra pdf enables PDF export
uv run streamlit run app/streamlit_app.py
```

Without `uv`:

```bash
cd general-agent
python -m venv .venv
.venv/bin/pip install -e ".[ui,pdf]"
.venv/bin/streamlit run app/streamlit_app.py
```

To classify with your own OpenRouter key, export it in the same shell that starts
Streamlit and select OpenRouter in the interface:

```bash
export OPENROUTER_API_KEY="sk-or-..."
uv run streamlit run app/streamlit_app.py
```

`.env` files are gitignored and are **not loaded automatically** — only exported
variables are visible. The interface binds to `127.0.0.1` only. It is a
single-user tool; do not expose it on a network. The `ui` extra is optional, so
the engine still installs and runs without Streamlit, and the OpenCode CLI is not
required to launch the interface or inspect results.

## Two flows, one engine

The default is a **three-step flow** for first-time users:

1. **Upload** — the observations and a separate human-labelled benchmark.
2. **Review** — the interface *proposes* the ID/text/label columns, category names
   and definitions, the label mapping, the model, and safe call/spend caps. Only
   genuine ambiguities are raised (two plausible text columns, labels that are
   not category names, numeric labels, mismatched or duplicate IDs, blank text, a
   missing credential). Everything else is collapsed under **Advanced settings**.
   Label polarity is never inferred: numeric or non-canonical labels always
   require an explicit mapping.
3. **Analyze & Download** — one checkbox authorizes the whole bounded run:
   validate → seal → five-row pilot → full blind classification → evaluation →
   disagreements → qualitative explanations (only when at least one disagreement
   exists) → statistical report → **study report PDF**. Progress, calls used,
   provider cost, **Stop**, **Resume**, and the downloads all live on this screen.
   The five-row pilot is approved automatically when it parses cleanly; if any
   pilot row fails, the run pauses with those rows and asks.

The original **seven-step flow** stays available behind the **Advanced mode**
button in the sidebar, for anyone who wants to control each stage.

One authorization covers the run. Changing the model or the study after
authorizing invalidates it and asks again; raising a cap also needs a new
authorization. The engine's safeguards are unchanged: the benchmark is sealed
before classification, the pilot approval binds the model and prompt hash,
predictions are frozen before evaluation, and caps are counted from the attempt
logs.

## Terms

- **Dataset** — observations to classify: an ID column and a text column.
- **Benchmark** — a separate file of human labels with the same IDs.
- **Seal** — stage 02 splits the files so the classifier can only read the ID and
  the text. The human labels are hidden until evaluation.
- **Pilot** — the first five observations, classified and reviewed before the
  full run.
- **Disagreement** — the AI answer differs from the human label. It is not a
  confirmed model error.

## First ten minutes (Advanced mode)

The seven-stage path below is the one shown in **Advanced mode**; the default
three-step flow is described under *Two flows, one engine* above.

1. **Data** — name the study, upload both files.
2. **Columns** — choose the ID, text, and human-label columns.
3. **Categories** — define the categories and map each human-label value to one.
4. **Model and limits** — choose an approved provider and model, set the call and
   spend caps, and acknowledge that text is sent to the provider.
5. **Validate and seal** — run the audit, fix anything it flags, then seal the
   benchmark and inspect the exact prompt the AI will receive.
6. **Classify** — run the five-row pilot, approve it, then classify in batches.
7. **Results** — evaluate, read the metrics, and download the report,
   predictions, and result tables.

## Qualitative analysis of disagreements

After evaluation, the Results page offers a **Qualitative analysis (AI
hypotheses)** section. It asks a model for a short, structured hypothesis about
each sampled AI–human disagreement, and for a small control sample of
agreements, then counts recurring reasons and writes a narrative report.

- The explanations are **hypotheses**, not confirmed causes and not proof of a
  model error; the interface and the report say so.
- It runs only after predictions are frozen and never changes them.
- It sends the observation text and the human label to the provider; the
  setup form requires an acknowledgement.
- A three-item **pilot** (two disagreements and one agreement) is explained
  first; the full run stays locked until the pilot is approved.
- Sampling is seeded; the control sample is drawn separately. Changing the
  settings after explanations exist is blocked, so a new study is required.
- A separate call and spend cap applies; the provider-reported cost is shown,
  with a character-based floor as an estimate.

Outputs are under `runs/<run_id>/qualitative/`: `tables/`, `explanations.csv`,
`narrative_report.md`, the attempt log, the approval, and the settings. From the
command line:

```bash
uv run python scripts/08_qualitative.py --config configs/my_study.yaml \
    --settings runs/<run_id>/qualitative/settings.yaml --pilot
# review, then
uv run python scripts/08_qualitative.py --config ... --settings ... --approve
uv run python scripts/08_qualitative.py --config ... --settings ... --run
uv run python scripts/08_qualitative.py --config ... --settings ... --report
```

`--report` makes no model call. `configs/qualitative_template.yaml` documents the
settings; `configs/qualitative_schema.json` validates them.

## Reports and PDF export

Three separate reports are available, all composed from the existing result
tables (no model call, no statistic recomputed):

1. **Per-study report** — one unique document per study: statistical results,
   and the qualitative hypotheses when stage 08 has run. If it has not, the
   report says so and includes no explanations.
2. **Qualitative report** — the stage 08 narrative, exported to PDF.
3. **Cross-dataset comparison** — the preserved stage 07 report, as a separate
   document.

In the interface:

- **7 · Results → Full report**: *Generate full report*, then *Download Full
  Report (PDF)* or the Markdown version. A qualitative PDF appears when the
  narrative exists.
- **Case studies**: each case study has *Generate full report (PDF)* /
  *Download Full Report (PDF)*, and the page has *Generate comparison report
  (PDF)*.

PDFs are written to the gitignored `reports/<run_id>/` (and
`reports/comparison/`), so `runs/` and `comparison/` are never modified. From the
command line:

```bash
uv run python scripts/09_report.py --config configs/my_study.yaml
uv run python scripts/09_report.py --run-id imdb_case_study
uv run python scripts/09_report.py --all
uv run python scripts/09_report.py --comparison
uv run python scripts/09_report.py --config configs/my_study.yaml --no-pdf
```

PDF export uses ReportLab (the `pdf` extra); without it, reports are Markdown
only and the interface shows the install hint. Figures are re-rendered as PNG
from the same result CSVs.

## Providers and credentials

The provider and model list is `configs/ui_providers.yaml`. The interface only
offers models listed there and never accepts a key. For an `openai_compatible`
provider it reports whether the named environment variable is present, never its
value. To add a provider, copy a block in that file and set `enabled: true`.

Two providers are enabled:

- **OpenRouter (your own API key)** — `openai_compatible`, `base_url:
  https://openrouter.ai/api/v1`, `api_key_env: OPENROUTER_API_KEY`. This is the
  **default selection** in the Provider dropdown on a fresh session and works
  with no OpenCode account. Export the key in the shell that launches the
  interface (a `.env` file is not read automatically) and reload; if the key is
  missing, the screen explains how to set it.
- **OpenCode CLI (local)** — `opencode_cli`; uses the signed-in local CLI, which
  must be installed and signed in. Select it manually in the Provider dropdown.

Because the default is OpenRouter, the OpenCode CLI is not required to reach the
first screen or inspect results.

Temperature is not controlled: the OpenCode CLI has no temperature flag, so it is
recorded as `not_requested` and the interface does not claim otherwise.

## Where a study lives

```text
runs/<run_id>/
  study.yaml        the generated engine configuration
  inputs/           unmodified copies of the uploaded files
  blind/            blind.csv (id, text only)
  sealed/           benchmark.csv (locked during classification)
  pilot/            pilot predictions, report, approval
  logs/             one JSONL record per model call
  predictions.csv   one row per observation
  results/          stage 05-06 tables
  qualitative/      stage 08 selection, explanations, tables, narrative
  report/           the stage 07 statistical report, tables, figures
  manifest.json     provenance accumulated across stages

reports/<run_id>/   composed per-study report (Markdown and PDF), gitignored
reports/comparison/ the cross-dataset comparison PDF, gitignored
```

Everything under `runs/` and `reports/` is gitignored.

## Tests

```bash
uv run pytest
```

Tests never make a model call: they use the stub provider. The interface tests
cover configuration building, the stage adapter, results reading, the leakage
boundary, read-only case studies, a headless render of the app, and the report
generation and PDF download path.

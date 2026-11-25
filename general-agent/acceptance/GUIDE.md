# Manual acceptance test — Streamlit interface

A five-minute browser walkthrough using two small synthetic files. Everything in
this kit is fictional (the "Aster Home" product line does not exist). No model
call happens until **you** authorize the run.

The interface opens in the default **three-step (Simple) flow** — **Upload →
Review → Analyze & Download** — and Part 1 walks through it. The original
seven-step procedure is kept, unchanged in spirit, behind **Advanced mode** in
Part 2.

## Files

| Purpose | File | Columns | Rows |
|---|---|---|---|
| Observations to classify | `acceptance/observations.csv` | `ticket_id`, `observation`, `channel` | 22 |
| Human benchmark | `acceptance/human_labels.csv` | `ref_id`, `human_label` | 22 |

- IDs match one-to-one: `AST-01` … `AST-22`.
- The benchmark labels are raw codes `POS` / `NEU` / `NEG`, so you must map them
  to categories (this is deliberate: it tests the mapping screen).
- `channel` is extra metadata; the AI never sees it.
- Some observations are deliberately mixed, sarcastic, or ambiguous
  (`AST-07`, `AST-10`, `AST-15`, `AST-20`). Expect some disagreements; that is
  the point of the test.
- The benchmark is a human reference, not proven truth.

## Before you start

From the repository root (`my-analysis/`):

```bash
cd general-agent
uv sync --extra ui --extra pdf --extra dev
uv run streamlit run app/streamlit_app.py
```

Open **http://127.0.0.1:8501**. The interface binds to localhost only. It starts
in **Simple mode**; Part 1 uses it.

Optional: export the credential named by your chosen provider before launching.

**No-cost preview (no model call).** Steps 1–5 of Part 2 — Data → Columns →
Categories → Model and limits → **Validate and seal** — run the audit and sealing
locally and make no model call. The five-row pilot and classification (the end of
Part 1, or Part 2 step 6) are the only stages that send text to a provider and may
incur cost; stop before them for a zero-call inspection.

---

# Part 1 — Simple mode (default, three steps)

## Step 1 · Upload

1. **Study name**: `manual_acceptance`.
2. **Dataset to classify**: upload `acceptance/observations.csv`.
   Expect: *22 rows · 3 columns · csv*.
3. **Human benchmark**: upload `acceptance/human_labels.csv`.
   Expect: *22 rows · 2 columns · csv*.
4. Click **Continue to review**.

## Step 2 · Review

Everything is proposed from the two files. Only the items that need you are
raised; the rest is collapsed under **Advanced settings**.

1. **Proposed plan**: confirm the observation ID (`ticket_id`), the observation
   text (`observation`), the benchmark ID (`ref_id`), and the human label
   (`human_label`).
2. **Categories and mapping**: the interface proposes category names taken from
   your labels (for these files, slugs such as `neg`, `pos`, `neu`) with
   **placeholder definitions**. Rename the categories to `negative`, `neutral`,
   `positive`, paste the definitions below, and map the raw values.

   | Category | Definition to paste |
   |---|---|
   | `negative` | The customer's overall assessment of the product or service is unfavourable. |
   | `neutral` | The customer reports information or a question without a clear favourable or unfavourable assessment. |
   | `positive` | The customer's overall assessment of the product or service is favourable. |

   | Raw value | Rows | Map to |
   |---|---|---|
   | `'POS'` | 7 | `positive` |
   | `'NEG'` | 8 | `negative` |
   | `'NEU'` | 7 | `neutral` |

   Label polarity is never inferred; every value must be mapped explicitly.
3. **Model and budget**: choose the provider and model, set the classification
   call cap to `40` (minimum needed is 5 pilot + 22 rows = 27) and the spend cap
   to `0.10`.
4. The screen shows a block while a category definition is still the
   placeholder, and says *Everything needed is ready* once it is not.
5. Click **Continue to analyze**.

## Step 3 · Analyze & Download

1. Read **What will run** and the **Budget** table; classification and the
   optional qualitative explanations are shown separately, with a combined
   maximum.
2. Tick the authorization checkbox and click **Authorize and run**. This is the
   only authorization for the bounded run; nothing is sent anywhere before it.
3. The screen drives the whole pipeline. The **five-row pilot is approved
   automatically** once all five rows parse to a declared category. If a pilot
   row fails, the run pauses and shows those rows with **Continue anyway**,
   **Re-run the pilot**, and **Stop**.
4. When the run reports success, download the report and the result files on the
   same screen.

## Verifying the seal yourself (Simple mode)

The Simple flow seals the benchmark and classifies during the authorized run. To
check the safeguard directly, open the run's files:

- `runs/manual_acceptance/blind/blind.csv` contains only `id` and `text`.
- `runs/manual_acceptance/sealed/benchmark.csv` holds the human labels; it is
  `chmod 000` while classification runs.
- A prompt log under `runs/manual_acceptance/logs/` contains the observation text
  and the category definitions, and must **not** contain a human label or
  `POS`/`NEG`/`NEU`.

---

# Part 2 — Advanced mode (the seven-step procedure)

Click **Advanced mode** in the sidebar to switch. The steps are Data → Columns →
Categories → Model and limits → Validate and seal → Classify → Results. The
engine and its safeguards are the same as in Simple mode; only the controls are
exposed stage by stage.

## Step 1 — Data

1. In the sidebar, click **Start a new study**.
2. **Study name**: `manual_acceptance`.
3. **Dataset to classify**: upload `acceptance/observations.csv`.
   Expect: *22 rows · 3 columns · csv*.
4. **Human benchmark**: upload `acceptance/human_labels.csv`.
   Expect: *22 rows · 2 columns · csv*.
5. Click **Continue to columns**.

## Step 2 — Columns

1. Dataset **ID column**: `ticket_id`.
2. Dataset **Text column**: `observation`.
3. Benchmark **ID column**: `ref_id`.
4. Benchmark **Human label column**: `human_label`.
5. You should see: **"22 IDs match in both files."**
6. Click **Continue to categories**.

## Step 3 — Categories

Add three categories (leave **"These categories have a natural order"** off):

| Category | Definition to paste |
|---|---|
| `negative` | The customer's overall assessment of the product or service is unfavourable. |
| `neutral` | The customer reports information or a question without a clear favourable or unfavourable assessment. |
| `positive` | The customer's overall assessment of the product or service is favourable. |

Then map the raw human labels:

| Raw value | Rows | Map to |
|---|---|---|
| `'POS'` | 7 | `positive` |
| `'NEG'` | 8 | `negative` |
| `'NEU'` | 7 | `neutral` |

- Under **If IDs do not match**, keep **"Stop and let me fix the files"**.
- The unmapped warning should disappear once all three are mapped.
- Click **Continue to model and limits**.

## Step 4 — Model and limits

1. **Provider** and **Model**: choose from the list. Check the credential line
   reads *available*; if it says missing, export the environment variable in the
   shell that started Streamlit and reload.
2. **Sample**: *All rows (22)*.
3. **Random seed**: keep `20261116`.
4. **Maximum calls**: `40` (minimum needed is 5 pilot + 22 rows = 27).
5. **Maximum spend (USD)**: `0.10`.
6. Tick **"I understand that the observation text will be sent to this
   provider."**
7. Click **Continue to validation**.

## Step 5 — Validate and seal (no model calls)

1. Optionally open **Prompt and report details** and set
   **Task role**: `a sentiment classifier for customer feedback`.
2. Click **Create study and validate**.
   Expect: checks **PASS**, 1 warning at most, and a row-accounting table.
3. Click **Seal benchmark and continue**.
   Expect: *Benchmark sealed. Human labels are hidden until evaluation.*
4. Scroll to **Exactly what the AI will receive**, pick a row, and read the exact
   prompt. It must contain the observation text and the category definitions, and
   must **not** contain the human label or `POS`/`NEG`/`NEU`.
5. The **Columns** and **Categories** steps are now locked in the sidebar.
6. Click **Continue to classification**.

## Step 6 — Classify (this is where live calls happen)

In Advanced mode the pilot gate is **manual**. Only continue when you accept the
spend. Nothing here is automatic.

1. Click **Run pilot (5 calls)**. Wait, then review the five rows: you see the
   ID, a text excerpt, the AI answer, and status — **never** the human label.
2. If they look sane, add an **Approval note** (e.g. `5/5 parseable`) and click
   **Approve pilot**.
3. Use **Run next 25 rows** (or **Run all remaining**). Watch the progress bar
   and the call counter stay within the cap.
4. When it says *All rows classified*, click **Continue to results**.

Expected: about 27 calls; roughly $0.01 at typical catalogue prices.

## Step 7 — Results

1. Check the headline metrics (accuracy, balanced accuracy, Cohen's kappa with
   intervals), the confusion matrix (rows = human, columns = AI), per-category
   results, and the disagreement table. Disagreements are shown as
   **unconfirmed**, never as proven errors.
2. Optional — **Qualitative analysis**: pick a model and sizes (15
   disagreements are possible here, so try 5 and 3), tick the consent box, click
   **Explain a 3-item pilot**, review, then **Approve pilot and explain the
   rest**, then **Generate qualitative report**. This is additional spend; skip
   it if you only want the statistical path.
3. Click **Generate full report**, then **Download Full Report (PDF)**. The
   report should contain the executive summary, the statistical sections, and
   (if you ran it) the qualitative findings — with no local file paths.
4. Open **Case studies** in the sidebar: the shipped IMDb and Financial
   PhraseBank studies should render read-only, unchanged.

---

## What the run creates

```text
runs/manual_acceptance/          inputs, blind input, sealed benchmark, predictions, results
reports/manual_acceptance/       study_report.md, study_report.pdf, qualitative_report.pdf
```

Both are gitignored, so they are not carried by a commit; keep them if you need
them after the test. Delete the `runs/manual_acceptance` and
`reports/manual_acceptance` folders to start over.

## Checklist

- [ ] Both files upload and preview.
- [ ] The Review screen proposes the columns, categories, and mapping.
- [ ] All three label values map; the placeholder definitions are replaced.
- [ ] Authorization runs the pipeline; a clean pilot is approved automatically.
- [ ] (Advanced) the pilot is reviewed and **approved manually** before the full
      run.
- [ ] Full run stays within the call cap.
- [ ] Results show metrics, confusion matrix, and disagreements.
- [ ] The blind input contains no labels; prompts contain no human label.
- [ ] Full report PDF downloads and looks complete.
- [ ] Shipped case studies still open unchanged.

## Notes

- This kit makes **no** model calls on its own; only the authorized run (and the
  optional qualitative run) can.
- If a credential is missing, validation and sealing still work; only
  classification is blocked.
- The synthetic labels are one plausible human reading, chosen to include a few
  genuinely arguable cases, so a handful of disagreements is expected and is not
  a failure of the interface.

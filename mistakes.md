# Mistakes, Limitations, and Unresolved Issues

**Project:** Can AI Really Understand Movie Reviews? (Business Analytics)
**Compiled:** 2026-11-16
**Scope:** Workflow run (`workflow_*`), Agent run (`agent_*`), and independent
Verification run (`verification_*`).

> **Document control.** `mistakes.md` was empty (0 bytes) before this entry, so
> no existing content was removed or overwritten. This review read the existing
> logs, tables, and reports but did not modify any dataset, analysis script, or
> result file, and made no model classification calls.
>
> **Evidence rule.** Every claim below cites a file that exists in the
> repository. Where a value was corrected before the final artifacts were
> written, the faulty intermediate value is not always preserved; those cases
> are marked **[value not retained]** and are not reconstructed or invented.

---

## How to read this document

Not every problem is the same kind of problem. Three categories are used:

| Category | Meaning | Count here |
|---|---|---|
| **Actual mistake** | A computation, statistic, table, or report statement was wrong and should not have been published as-is. | 10 (M1–M10; M10 is a minor presentation defect) |
| **Documentation / provenance issue** | The analysis result is correct, but the files describing or feeding it are stale, ambiguous, or contradictory. | 3 (D1–D3) |
| **Limitation / unresolved issue** | A known constraint, or a planned check that was never done. Not an error in the arithmetic. | 6 (U1–U6) |

A separate section lists **differences between runs that are legitimate
methodological choices, not mistakes**, so they are not miscounted as errors.

---

# Part A — Actual mistakes

## Summary table

| ID | Run | Mistake | Detected by | Status |
|---|---|---|---|---|
| M1 | Workflow | Invalid odds ratio for `complex` under complete separation | Implausible OR = 0, CI ≈ 10^90, coefficient ≈ 10^8 | Corrected (reported `not_estimable`) |
| M2 | Agent | Cohen's κ reported as 0.905 instead of 0.850 | Recomputing κ from the 95/90/8/7 confusion matrix | Corrected in final artifacts |
| M3 | Agent | `stats.optimize.minimize` (wrong SciPy namespace) | Python failure | Corrected |
| M4 | Agent | Report generator `KeyError` for `straightforward` | Python failure | Corrected |
| M5 | Agent | Consensus confidence interval hard-coded from console output | Code/report review | Corrected (now read from a table) |
| M6 | Agent | Confusion between "manual error" and "manual–label dispute" | Early exploration; three-way decomposition | Conceptually corrected |
| M7 | Agent | McNemar discordant cells mis-tabulated (10/5 instead of 1/5) | Recomputing the 2×2 tables in verification | **Corrected 2026-11-16** (see correction record, Part F) |
| M8 | Verification | Cohen's κ standard error inflated (0.0334 vs 0.0242) | Bootstrap + Monte-Carlo cross-check | Corrected |
| M9 | Verification | Sensitivity filter variable reused; per-category `n` wrong (e.g. 6 vs 27) | Row-count reconciliation | Corrected |
| M10 | Agent | Figure label/legend overlap | Visual inspection | Corrected (minor) |

---

## M1 — Invalid odds ratio for the `complex` category

**Evidence:** `results/workflow_40_summary.md` §10 (lines 318–368);
`results/workflow_30_t05_adjusted_logistic.csv`.

1. **What went wrong.** The first execution of `src/workflow_30_analysis.py`
   reported, for the `complex` term in the overlap-adjusted logistic regression:

   ```
   complex   coef = -1.4e+8-ish   OR = 0.000   upper 95% CI ≈ 2.2e90
   ```

   This is not a real estimate. An odds ratio of exactly zero with a confidence
   interval of order 10^90 cannot come from a sample of 200 reviews.

2. **Why it happened.** `complex` contained **zero errors in 20 reviews**
   (0/20). The maximum-likelihood coefficient is driven to −∞ — **complete
   separation**. The ridge penalty kept the fit numerically finite, so the code
   returned a number instead of failing, but the number had no statistical
   meaning.

3. **How it was detected.** Three independent symptoms in the first output:
   (a) an OR of exactly 0 for a category with no plausible protective mechanism;
   (b) a CI of order 10^90; (c) a coefficient magnitude (~10^8) incompatible with
   n = 200. Checking the raw cell confirmed 0/20 errors. The verification run
   independently reproduced the same separation with an ordinary MLE
   (`coef = −23.556`, `SE = 94,303` in `results/verification_03_logistic.csv`),
   confirming the diagnosis.

4. **How it was corrected.** A separation guard was added to the **reporting
   step**: any term with |coefficient| > 8 or SE > 8 is reported as
   `not_estimable` with an explicit note. The corrected table
   (`results/workflow_30_t05_adjusted_logistic.csv`) now says
   `complex, not_estimable, ..., complete separation (no errors in this cell)`.
   The descriptive results for `complex` were unaffected (0/20; risk difference
   −0.019, 95% CI −0.066 to 0.143; Fisher p = 1.000). The verification run used
   Firth penalisation instead, which returns a finite but non-significant
   coefficient (−1.403, p = 0.338).

5. **What this teaches about relying on AI agents.** An AI agent will readily
   print a finite number produced by a degenerate model. Fluent, numerically
   formatted output is not evidence that the estimate is valid. Automated
   boundary checks (zero cells, separation, absurd standard errors) and an
   explicit `not_estimable` state are needed. The agent did eventually catch
   this, but only after the first result was produced and inspected.

---

## M2 — Cohen's κ reported as 0.905 instead of 0.850

**Evidence:** correct value in `results/agent_02_accuracy_log.txt` line 25
(`manual_vs_imdb_label ... kappa=0.8499`), `results/agent_report.md` line 87,
and `results/workflow_30_t02_agreement_kappa.csv`
(`human_vs_imdb_kappa,0.850`). The manual-vs-IMDb confusion matrix is
TP 95 / TN 90 / FP 8 / FN 7.

1. **What went wrong.** The Cohen's κ for **manual annotation vs IMDb label**
   was initially reported as **0.905**. The correct value is **0.850**.
   **[value not retained]**: the 0.905 intermediate output is not present in any
   retained file; only the corrected artifacts survive. This entry therefore
   records the error as reported in the project record, and does not invent a
   mechanism for it.

2. **Why it happened.** Not recorded in the retained logs, so I am not sure.
   The correct κ is 0.8499, which is distinct from κ(AI, IMDb) = 0.8899 and
   κ(AI, manual) = 0.9399, so the error was not simply a mislabeled rater pair
   from the final tables. A mis-specified contingency table or an intermediate
   formula error is the plausible family of causes, but the exact cause cannot
   be established from the files.

3. **How it was detected.** Recomputing κ from the published 2×2 table:
   95/90/8/7 gives observed agreement 0.925 and expected agreement 0.501,
   hence κ = 0.8499. The final log (`agent_02_accuracy_log.txt` line 25) and the
   workflow table both report 0.850.

4. **How it was corrected.** The final Agent metrics, the report table, and the
   Workflow table all use 0.850. No retained artifact still shows 0.905.

5. **What this teaches about relying on AI agents.** LLMs can state a
   plausible-looking metric that is not tied to any generated table. The
   AGENTS.md rule "every number comes from a file in `results/`" exists
   precisely for this failure mode. Key metrics should be re-derived from the
   confusion matrix, not copied from prose.

---

## M3 — Wrong SciPy namespace in the shared Agent utility

**Evidence:** `results/agent_autonomous_decisions.md` §5, E1 (lines 282–286).

1. **What went wrong.** The first version of `agent_00_utils.py` called
   `stats.optimize.minimize`. `scipy.stats` does not expose `optimize`, so the
   Firth model code could not run.

2. **Why it happened.** Two SciPy submodules were conflated:
   `scipy.stats` and `scipy.optimize`.

3. **How it was detected.** The Python call failed (the decision log records the
   bad namespace and the correction).

4. **How it was corrected.** `optimize` was imported from SciPy and the call was
   changed to `optimize.minimize`. The Firth models were re-run and all
   converged.

5. **What this teaches about relying on AI agents.** Agents make ordinary
   import/namespace slips. Code that looks syntactically plausible still has to
   be executed; "the agent says the model converged" is not a substitute for a
   run log and convergence diagnostics.

---

## M4 — `KeyError` for `straightforward` in the Agent report generator

**Evidence:** `results/agent_autonomous_decisions.md` §5, E2 (lines 288–294).

1. **What went wrong.** The report generator indexed
   `tl.loc['straightforward', ...]`, but the compact
   `results/agent_03_category_tests.csv` has no row for the reference class —
   the reference values live in columns on the flag rows. The report crashed
   with a `KeyError`.

2. **Why it happened.** The code was written against an imagined table schema
   rather than the schema the analysis script actually emitted.

3. **How it was detected.** The report-generation run raised `KeyError` for
   `straightforward`.

4. **How it was corrected.** The generator was changed to use the reference
   columns from the flag rows plus the analysis dataset for reference n/errors.

5. **What this teaches about relying on AI agents.** Agents often write code
   against a remembered or assumed schema. Emitting and validating a schema
   contract between pipeline stages, then running the whole pipeline end to end,
   catches this class of failure.

---

## M5 — Hard-coded consensus confidence interval in the first report draft

**Evidence:** `results/agent_autonomous_decisions.md` §5, E3 (lines 296–301);
the replacement table `results/agent_02_consensus_summary.csv` now exists.

1. **What went wrong.** The consensus-accuracy confidence interval was typed
   directly into the report generator from the step-2 console output instead of
   being read from a results table. The same shortcut affected the "AI sides
   with manual" interval.

2. **Why it happened.** It was faster to paste the console number than to add a
   machine-readable output file. This bypassed the project rule that every
   reported number must trace to `results/`.

3. **How it was detected.** During review of the report generator: the value in
   the prose had no corresponding table cell.

4. **How it was corrected.** `results/agent_02_consensus_summary.csv` was added
   to `agent_02_accuracy.py`, and the report now reads both intervals from it.

5. **What this teaches about relying on AI agents.** Hard-coded numbers are
   silent single-source-of-truth violations. They survive re-runs and can drift
   away from the data without any error message. The fix is structural: generate
   the report *from* the result tables, never from console output.

---

## M6 — Confusion between "manual error" and "manual–label dispute"

**Evidence:** `results/agent_autonomous_decisions.md` §5, E5 (lines 311–317);
the three-way decomposition in `results/agent_02_three_way_agreement.csv` and
`results/agent_report.md` lines 100–111.

1. **What went wrong.** During early exploration, the Agent briefly conflated
   two different quantities:
   - **15** reviews where the manual annotation disagrees with the **IMDb
     label**; and
   - **6** reviews where the AI prediction disagrees with the **manual
     annotation**.
   These are different row sets and different questions.

2. **Why it happened.** Three 2×2 tables share the same 200 rows (AI vs label,
   manual vs label, AI vs manual). Without explicit cell labels, it is easy to
   move a count from one table into another.

3. **How it was detected.** During early exploration, the counts did not
   reconcile: 15 manual–label disputes plus 6 AI–manual errors cannot both be
   "the number of AI mistakes". The three-way decomposition made the overlap
   explicit.

4. **How it was corrected (conceptually).** The scripts now compute
   `error_vs_manual` and `manual_disagrees_with_label` as separate variables,
   and the report uses the four-way decomposition
   (all agree / AI disagrees with both / dispute–AI sides with label /
   dispute–AI sides with manual). The interpretation now correctly states that
   5 of the 6 manual-reference "errors" are rows where manual and IMDb disagree
   and the AI sides with IMDb.

5. **What this teaches about relying on AI agents.** A conceptual confusion can
   be narrated correctly in prose while still being present in code. The agent
   wrote about the distinction and *also* carried it into a numeric table (see
   M7). Prose self-awareness is not a guarantee that the computation is clean.

---

## M7 — McNemar discordant cells mis-tabulated (10/5 instead of 1/5) — CORRECTED 2026-11-16

**Status:** **Resolved on 2026-11-16.** The original wrong output is preserved in
this entry as evidence, and the correction is recorded in Part F.

**Evidence (original mistake):** `results/agent_report.md` line 111 (before the
fix); `results/agent_02_paired_tests.csv` row
`mcnemar_ai_vs_manual_errors_vs_label` (before the fix);
`results/agent_02_accuracy_log.txt` (before the fix).

**Evidence (correction):** `src/agent_02_accuracy.py` (McNemar block, corrected);
`results/agent_02_paired_tests.csv` (after the fix);
`results/agent_02_accuracy_log.txt` (after the fix);
`results/agent_report.md` line 111 (after the fix); Part F below.

1. **What went wrong.** The Agent's McNemar test comparing AI and manual error
   rates against the IMDb label was reported as **p = 0.302**, with
   **"AI-only errors 10, manual-only errors 5"**. The correct discordant counts
   are **AI-only errors = 1** and **manual-only errors = 5**, giving exact
   McNemar **p = 0.21875**. The value 10 is the **both-wrong** count, not an
   AI-only error.

   Original (wrong) output, retained as evidence:

   ```
   AI-only errors 10, manual-only errors 5, exact McNemar p = 0.301758
   ```

   Correct output:

   ```
   AI-only errors  1, manual-only errors 5, exact McNemar p = 0.218750
   ```

   Reconciliation (verification, from the joined data):
   AI errors vs IMDb = 11; manual errors vs IMDb = 15; both wrong = 10;
   AI-only = 11 − 10 = **1**; manual-only = 15 − 10 = **5**; neither = 184.
   A binomial test on 1 vs 5 gives p = 0.21875; a binomial test on 10 vs 5 gives
   p = 0.301758, which is exactly the originally reported p — confirming which
   cells were used.

2. **Why it happened.** Root cause identified during the correction:
   `error_vs_manual` in `results/agent_01_analysis_dataset.csv` is defined as
   `(ai_sentiment != sentiment)` — an **AI-vs-manual disagreement**, not a
   **manual-vs-IMDb error**. The script therefore cross-tabulated
   `error_vs_label` × `error_vs_manual`:

   ```
   [[184,  5],
    [ 10,  1]]
   ```

   In that table, `loc[1, 0] = 10` is "AI error **and** AI agrees with manual"
   (i.e. both are wrong against IMDb), and `loc[0, 1] = 5` is "AI correct and
   AI differs from manual" (manual-only error). The 10 was then read as
   "AI-only error", which is the M6 conceptual confusion expressed numerically.
   The correct table uses manual error vs **IMDb** on both margins:

   ```
   [[184,  5],
    [  1, 10]]
   ```

3. **How it was detected.** During the independent verification, the 2×2 tables
   were rebuilt from the raw joined data. The reported 10/5 pair did not
   reconcile with the known totals (AI errors 11, manual errors 15, both 10).
   For the correction, the counts were recomputed with four independent checks,
   all agreeing:
   - direct cell counts from `data/derived/workflow_joined.csv`;
   - `scipy.stats.binomtest(1, 6, 0.5)` → p = 0.21875;
   - `statsmodels.stats.contingency_tables.mcnemar(..., exact=True)` → p = 0.21875;
   - manual enumeration of the two-sided binomial tail → p = 0.21875.
   The wrong 10/5 table reproduces exactly p = 0.301758, confirming the
   diagnosis.

4. **How it was corrected.** The Agent pipeline was repaired end to end:
   - `src/agent_02_accuracy.py`: replaced the wrong second classifier with
     `manual_error_vs_label = (df.sentiment != df.label_sentiment)` and added two
     reconciliation assertions (each margin must equal its total).
   - Re-ran `src/agent_02_accuracy.py`: `results/agent_02_paired_tests.csv` now
     reports discordant 1 and 5, statistic 1.0, p = 0.218750; the log is updated.
   - Re-ran `src/agent_06_report.py`: `results/agent_report.md` now reads
     "p = 0.219 (AI-only errors 1, manual-only errors 5)".
   - All other Agent outputs were byte-identical (hash-verified); the report diff
     contained only the McNemar sentence. No dataset, Workflow output, or model
     call was touched. Full record in Part F.

5. **What this teaches about relying on AI agents.** An agent can detect a
   conceptual problem, document it, and still leave the corresponding numeric
   error in the output. Self-documentation is not self-correction. Independent
   recomputation of every paired table — and a totals reconciliation
   (b + c must equal the discordant total) — is necessary. The durable fix is
   structural: name the variable after the comparison it represents and assert
   the marginal totals, so a wrong-cell substitution fails loudly.

---

## M8 — Verification: inflated Cohen's κ standard error

**Evidence:** `results/verification_findings.md` §7 item 1;
final corrected value in `results/verification_02_agreement.csv`
(`kappa_se_asymptotic = 0.02416`); method note in
`src/verification_utils.py::kappa_se`.

1. **What went wrong.** The verification's first closed-form (Fleiss-style)
   κ standard error was **0.0334**, about **38% larger** than the correct
   ≈0.0242. This made the asymptotic confidence interval too wide.

2. **Why it happened.** The closed-form variance implementation was incorrect
   for these data (the formula/term handling did not match the actual sampling
   distribution). The point estimate κ = 0.9399 was correct; only the SE was
   wrong.

3. **How it was detected.** The analytic SE was cross-checked against two
   independent methods:
   - a 20,000-draw percentile bootstrap (SE ≈ 0.0243), and
   - a 20,000-draw Monte-Carlo simulation from the observed multinomial
     distribution (SE ≈ 0.0242).
   The 0.0334 value was outside both.

4. **How it was corrected.** The closed form was replaced by a numerical
   multinomial delta method. The new method returns **0.02416**, matching the
   bootstrap, and the final table reports the bootstrap CI 0.8895–0.9800. The
   failed method is recorded in the findings rather than silently deleted.

5. **What this teaches about relying on AI agents.** An "independent"
   verification can also contain a subtle formula bug. Analytic standard errors
   should be validated against resampling when n is small and cells are
   unbalanced. AI agents produce confident-looking SEs and CIs even when the
   underlying variance formula is wrong.

---

## M9 — Verification: sensitivity filter variable reused, per-category `n` wrong

**Evidence:** `results/verification_findings.md` §7 item 2;
corrected `results/verification_03_sensitivity.csv` (ambiguous n = 27,
mixed n = 44, sarcasm n = 23, complex n = 15, straightforward n = 106);
matching prior table `results/workflow_30_t06_sensitivity.csv`.

1. **What went wrong.** In the first verification draft, the per-category
   "exclude `uncertainty_sentiment = 2`" sensitivity rows actually used the
   stricter "exclude any uncertainty ≥ 1" filter. For example, `ambiguous` had
   **n = 6** instead of **n = 27**, and `mixed` had 32 instead of 44.

2. **Why it happened.** A single mutable variable `keep` was assigned twice.
   The second assignment (the stricter filter) overwrote the first, and the
   per-category loop below used the overwritten value.

3. **How it was detected.** The subgroup `n`s did not reconcile: the sum of
   per-category reductions was 68, while the overall reduction from 200 to 187
   was only 13. The numbers also disagreed with the prior sensitivity table.

4. **How it was corrected.** Dedicated variables `keep_sent` and `keep_any`
   were introduced, with a code comment warning not to reuse the stricter one.
   Re-running produced ambiguous n = 27, matching the independent prior table.
   The corrected file is `results/verification_03_sensitivity.csv`.

5. **What this teaches about relying on AI agents.** State reuse is a common
   silent bug. Reconciliation checks (subgroup `n`s must sum to the overall
   analysis `n`; totals must match across tables) catch filter drift that no
   exception would ever raise. AI agents can overwrite a variable and continue
   producing plausible output.

---

## M10 (minor) — Figure label overlap in the Agent figures

**Evidence:** `results/agent_autonomous_decisions.md` §5, E4 (lines 303–309).

1. **What went wrong.** In Figure 2 the category labels collided; in Figure 3
   the point annotations overlapped the lines; in Figure 5 the legend covered
   the longest bar.
2. **Why it happened.** Default matplotlib sizing and annotation placement were
   not checked against the actual label lengths.
3. **How it was detected.** Visual inspection of the rendered figures.
4. **How it was corrected.** Figure 2 was widened, labels shortened, Figure 3's
   sample sizes moved to the axis ticks, and Figure 5's legend moved below the
   plot; all five figures were re-rendered and checked.
5. **What this teaches about relying on AI agents.** Agents can generate
   figures that are numerically correct but unreadable. Figures need a human
   visual check, not just a successful script exit.

---

# Part B — Documentation and provenance issues (not calculation errors)

These did **not** change any reported result. They are recorded because they
make the project harder to audit and could mislead a reader.

## D1 — Outdated model documentation in the frozen specification

**Evidence:** `src/workflow_00_model_and_prompt.md` line 15
(`Primary model | opencode/deepseek-v4.1-flash-free`) versus the same file's
v1.3 changelog (lines 164–172: the executed model is
`openrouter/deepseek/deepseek-v4.1-flash-1017`), and
`results/workflow_40_summary.md` §3.

- **What:** Section 1 of the frozen model/prompt document still names the
  free-tier OpenCode model as primary, while every analysed prediction comes
  from the OpenRouter model.
- **Why:** The run was restarted on a paid single endpoint after the free-tier
  quota was exhausted; the change was recorded in a changelog appended to the
  same file, but the top-of-document "Primary model" line was not updated.
- **Detection:** Comparing the model string in the joined data / run log with
  the model string in §1.
- **Status:** Not an error in the analysis (the changelog is authoritative and
  the joined data carry the correct model). It is a **stale summary line**. A
  reader who stops at §1 would describe the wrong model.
- **Lesson:** An agent will append a correction without rewriting the stale
  summary. Version-control-style "single source of truth" needs the summary
  fields updated, not just a changelog entry.

## D2 — Stale prediction file with an ambiguous name

**Evidence:** `data/derived/workflow_ai_predictions.csv`;
`data/derived/workflow_13_predictions_openrouter.csv`;
`results/workflow_40_summary.md` §3 provenance note (lines 72–76);
`src/workflow_00_model_and_prompt.md` v1.3 changelog.

- **What:** `workflow_ai_predictions.csv` is the abandoned free-tier run. It has
  200 rows but only **73** `ok` predictions, **126** `failed_all_attempts`, and
  **1** `quota_blocked`; **127** `ai_sentiment` cells are missing. Among the 73
  rows that do have a prediction, **2** disagree with the analysed OpenRouter
  predictions.
- **Why:** The free-tier quota was exhausted twice; the file was preserved for
  provenance when the run moved to OpenRouter.
- **Detection:** Comparing the file's `model` and `status` columns with the
  joined dataset.
- **Status:** Documented as unused, so **not an analysis error**. The hazard is
  the generic name: a future reader could mistake it for the analysed
  predictions. It should be renamed or clearly marked `DO_NOT_USE`.
- **Note on a verification imprecision:** `results/verification_06_log.txt`
  summarised this as "disagrees ... on 129/200 rows". That 129 is **127 missing
  cells + 2 genuine disagreements**, not 129 substantive model disagreements.
  The correct statement is the one above.
- **Lesson:** Agents preserve artifacts helpfully but do not necessarily give
  them self-describing names. Stale outputs should be quarantined, not left
  beside the real ones.

## D3 — Blank `annotation_timestamp`, with conflicting explanations

**Evidence:** `data/annotations.csv` (all 200 timestamps blank);
`results/06_transfer_annotations.txt` ("annotation_timestamp: left blank
(decision at plan approval)");
`results/agent_autonomous_decisions.md` E6 ("likely lost in the spreadsheet
export"); `codebook.md` line 182 (requires an ISO timestamp per row).

- **What:** The timestamp field is blank for all 200 rows, although the codebook
  specifies an ISO timestamp.
- **Why:** The record is contradictory: the transfer log says it was a
  deliberate decision at plan approval, while the Agent's decision log says it
  was probably lost in the spreadsheet export. The `.xlsx` header does not
  contain a timestamp column at all, which is more consistent with "never
  collected" than "lost in export". I am not sure which account is correct.
- **Detection:** Automated missing-value check in
  `results/verification_01_audit.txt` and the join log.
- **Status:** Unresolved documentation inconsistency. It does not affect any
  analysis (the field is unused), but the two explanations should be reconciled
  and the codebook either updated or the field collected.
- **Lesson:** When an agent documents a data gap, it may give a plausible cause
  that later contradicts another log. Gaps should be traced to the source file,
  not explained from memory.

---

# Part C — Limitations and unresolved issues (not mistakes)

These are constraints or planned checks that were not completed. They are
**not** arithmetic errors, and they are not "fixed" by re-running a script.

## U1 — Missing intra-annotator reliability check

**Evidence:** `codebook.md` §7 (lines 216–219: re-annotate ~20 reviews at least
one day later, same seed-logged selection);
`data/annotations.xlsx` has only the `blind_sample` sheet and an empty
`Foglio1`; `results/06_transfer_annotations.txt` notes the empty second sheet;
`results/agent_autonomous_decisions.md` E7; `results/workflow_40_summary.md`
§9.1 ("the planned intra-annotator re-check was not part of this run").

- **Status:** **Not done.** No reliability estimate exists.
- **Consequence:** The stability of the manual sentiment and of the four
  difficulty flags is unknown. This is a genuine limitation, not a mistake:
  no incorrect reliability number was reported, because no number was reported.
- **What should happen next:** Re-annotate ~20 seeded reviews and report simple
  agreement and κ. Until then, category results inherit unmeasured annotation
  noise.
- **Lesson:** An AI agent cannot perform a human re-annotation. It can only
  report that the check is missing — which it did, correctly. Do not let an
  agent's polished report create the impression that every planned safeguard
  was executed.

## U2 — Single annotator, no adjudication

**Evidence:** `codebook.md` §6–§7; `results/agent_report.md` line 310.

- One blind annotator, no second independent annotator, no adjudicated gold
  standard. Inter-annotator agreement cannot be computed.
- The manual reference is a second text-based reading, not "truth".

## U3 — Small errors count, low power, overlapping categories

**Evidence:** `results/verification_04_mde.csv`; `results/workflow_40_summary.md`
§9.3.

- Only 11 AI errors against IMDb (and 6 against the manual annotation).
- Minimum detectable differences at 80% power are ~19–29 percentage points
  against the 106 straightforward reviews.
- The four category tests reuse the same reference group and are dependent;
  correction choices (none / BH / Holm) change adjusted p-values.
- "Not significant" here means "underpowered", not "no effect".

## U4 — IMDb label provenance is unverified

**Evidence:** `data/README.md` §5.2; `results/workflow_40_summary.md` §9.2.

- IMDb labels come from star ratings, not from reading the text.
- 15/200 reviews have a manual label that disagrees with IMDb.
- Most apparent "AI errors" (5 of 6 manual-reference errors, and a large share
  of label errors) are on rows where the two human references disagree. The
  label-based error rate is therefore an upper bound on genuine model error,
  not a clean estimate.

## U5 — Uncontrolled temperature and unrecorded serving endpoint

**Evidence:** `src/workflow_00_model_and_prompt.md` §1 and §3;
`results/workflow_40_summary.md` §9.5–§9.6.

- Temperature was not settable through the access path.
- The upstream provider/endpoint is not recorded per call, so quantization may
  vary.
- No run-to-run stability re-run was performed, so reproducibility of the 200
  predictions is unmeasured.

## U6 — Free-tier quota exhaustion and model switch (operational constraint)

**Evidence:** `results/workflow_10_resume_log.txt`;
`results/workflow_12_validate_resume.txt`;
`results/workflow_13_validate.txt`;
`src/workflow_00_model_and_prompt.md` v1.2–v1.3.

- The free OpenCode proxy exhausted its quota at 35/200 and again at 73/200.
- The run was restarted on a single paid OpenRouter endpoint for all 200
  reviews (v1.3), rather than mixing endpoints.
- This is an **external constraint handled by a documented protocol change**,
  not an analysis mistake. It did, however, create the stale prediction file
  (D2) and the stale model summary (D1).

---

# Part D — Differences between runs that are not mistakes

These were flagged during comparison but are legitimate methodological choices.
They are listed so they are not mistaken for errors.

| Topic | Workflow | Agent | Verification | Why it is not an error |
|---|---|---|---|---|
| Multiplicity | No correction (descriptive) | Benjamini–Hochberg q | Holm–Bonferroni | All three are defensible; only `mixed` is borderline (q/Holm ≈ 0.055) under any choice |
| Adjusted logistic | Ridge 1e-4 | Firth | Firth | Ridge could not estimate `complex`; Firth returns a finite penalised estimate. Different estimators, same substantive conclusion |
| Trend test variance | — | N−1 convention (z = 1.3132) | N convention (z = 1.3165) | Immaterial; both p > 0.18 |
| κ standard error | Not reported | Closed form (0.0245) | Delta method (0.02416) | Both agree with the bootstrap; difference is < 2% of the estimate |
| Primary reference | IMDb label | IMDb label | IMDb label | Same choice; manual annotation reported as a secondary reference |

---

# Part E — What this project teaches about relying on AI agents

1. **Plausible output is not verified output.** The invalid `complex` OR (M1),
   the wrong κ (M2), and the wrong McNemar cells (M7) were all formatted like
   real results. Fluency and numerical formatting are not evidence.
2. **Agents can document a problem and still not fix it.** M6/M7 is the clearest
   case: the Agent wrote correctly about the disagreement types and still left
   the wrong cell in the McNemar table until it was independently recomputed and
   corrected on 2026-11-16.
3. **Silent state reuse is a recurring failure mode.** The verification's
   sensitivity bug (M9) produced perfectly plausible output with the wrong `n`.
   Reconciliation checks catch what exceptions do not.
4. **Boundary conditions need explicit handling.** Complete separation (M1) is
   a modelling boundary, not a coding exception. Agents need a rule that says
   "report `not_estimable`", or they will report a number.
5. **Hard-coded numbers are a governance failure.** M5 shows how easily a
   console value enters a report. The structural fix — report generated from
   `results/` tables — is more reliable than asking the agent to be careful.
6. **Self-verification is not independent verification.** The verification run
   found its own kappa-SE and sensitivity bugs (M8, M9). Independence is a
   process (separate code, separate outputs, compare last), not a property of a
   model.
7. **Agents are good at preserving artifacts, bad at labelling them.** The stale
   prediction file (D2) and stale model summary (D1) are helpful for provenance
   but dangerous for a reader. Names and status flags must make "unused" obvious.
8. **Human-only checks cannot be delegated.** The missing intra-annotator
   reliability check (U1) is the one item an AI agent could not complete. A
   polished report must not blur the line between "checked" and "not checked".

---

# Part F — Correction record: M7 McNemar (2026-11-16)

**Scope of the repair:** corrected the Agent script, the results table, the run
log, and the report sentence. No dataset (`data/`), Workflow output
(`results/workflow_*`), or model call was touched.

## F1 — Independent recomputation of the discordant counts

Source data: `data/derived/workflow_joined.csv` (read-only, 200 rows).
Definition: AI error = `ai_sentiment != imdb_label`; manual error =
`human_sentiment != imdb_label`. Both margins therefore use the **same**
reference (IMDb), which is what the McNemar test requires.

Direct 2×2 table (rows = AI error, columns = manual error):

```
              manual correct   manual error
AI correct         184              5
AI error             1             10
```

- **AI-only errors (b) = 1**
- **Manual-only errors (c) = 5**
- **Both wrong = 10**
- **Neither wrong = 184**
- **Exact McNemar p = 0.21875**

Four independent checks, all agreeing:

| Check | Result |
|---|---|
| Direct cell counts from the joined CSV | b = 1, c = 5 |
| `scipy.stats.binomtest(1, 6, 0.5)` (two-sided) | p = 0.21875 |
| `statsmodels...mcnemar(table, exact=True)` | p = 0.21875, statistic = 1.0 |
| Manual enumeration of the two-sided binomial tail | p = 0.21875 |

The original wrong table (10 vs 5) reproduces **p = 0.301758** exactly, which is
the value that appeared in the original report. That confirms the diagnosis:
the 10 was the both-wrong cell, not an AI-only error.

## F2 — Root cause and code fix

`results/agent_01_analysis_dataset.csv` defines:

- `error_vs_label = (ai_sentiment != label_sentiment)` — correct for the AI margin;
- `error_vs_manual = (ai_sentiment != sentiment)` — this is an **AI-vs-manual
  disagreement**, not a **manual-vs-IMDb error**.

The old code cross-tabulated `error_vs_label` × `error_vs_manual`:

```
[[184,  5],
 [ 10,  1]]
```

so `loc[1, 0] = 10` was read as "AI-only error" when it is actually "AI error
and AI agrees with manual" (both wrong against IMDb). The fix builds the second
margin explicitly:

```python
manual_error_vs_label = (df.sentiment != df.label_sentiment).astype(int)
tab_err = pd.crosstab(df.error_vs_label, manual_error_vs_label)
```

Two reconciliation assertions were added so the bug cannot silently return:
the AI-error margin must equal `df.error_vs_label.sum()`, and the manual-error
margin must equal `manual_error_vs_label.sum()`.

## F3 — Files changed

| File | Change | Before → after |
|---|---|---|
| `src/agent_02_accuracy.py` | Replaced `df.error_vs_manual` with `manual_error_vs_label`; added two margin assertions | wrong classifier → correct classifier |
| `results/agent_02_paired_tests.csv` | Regenerated by the corrected script | b 10 → 1; statistic 5.0 → 1.0; p 0.301758 → 0.218750; c unchanged at 5 |
| `results/agent_02_accuracy_log.txt` | Regenerated by the corrected script | same numeric update |
| `results/agent_report.md` | Regenerated by `src/agent_06_report.py` (which reads the table) | "p = 0.302 (AI-only errors 10…)" → "p = 0.219 (AI-only errors 1…)" |
| `mistakes.md` | This correction record; M7 marked resolved | status + evidence |

The other McNemar row (`mcnemar_ai_vs_manual_predictions`, 4 vs 2,
p = 0.6875) is unchanged.

## F4 — Proof that all other results are unchanged

A SHA-256 manifest of all 147 files under `data/`, `results/`, and `src/`
(excluding `__pycache__`) was taken **before** any change. After the correction,
exactly **4** files differ — the four intended files above — and **143** are
byte-identical.

In particular, all of the following are unchanged:

- every file in `data/` (including `data/raw/` and
  `data/derived/workflow_joined.csv`);
- every `results/workflow_*` file;
- the other Agent outputs: `agent_02_metrics.csv`,
  `agent_02_confusion_*.csv`, `agent_02_three_way_agreement.csv`,
  `agent_02_consensus_summary.csv`, `agent_02_error_direction.csv`;
- `src/agent_06_report.py` (the report template already read the table, so the
  sentence corrected itself when the table was fixed);
- all Verification outputs.

A line-level diff of `results/agent_report.md` before vs after shows **only**
the McNemar sentence changed.

## F5 — Effect on conclusions

None. The corrected test still gives p = 0.21875 > 0.05, so the substantive
statement is unchanged: there is no statistical evidence that the AI or the
manual annotation is closer to the IMDb labels. The correction fixes the
reported test statistic and counts, not the interpretation. The original wrong
values remain quoted in M7 and F1 as evidence.

*End of correction record.*

---

# Appendix — Evidence index

| File | Used for |
|---|---|
| `results/workflow_40_summary.md` §10 | M1 (invalid OR, separation, correction) |
| `results/workflow_30_t05_adjusted_logistic.csv` | M1 (corrected `not_estimable`) |
| `results/verification_03_logistic.csv` | M1 (independent reproduction of separation) |
| `results/agent_02_accuracy_log.txt` | M2 (κ 0.8499), M7 (error totals) |
| `results/agent_report.md` line 111 | M7 (original 10/5, p = 0.302; corrected 1/5, p = 0.219) |
| `results/agent_02_paired_tests.csv` | M7 (original 10/5; corrected 1/5) |
| `results/agent_autonomous_decisions.md` E1–E7 | M3, M4, M5, M6, M10, D3, U1 |
| `results/agent_02_consensus_summary.csv` | M5 (replacement table) |
| `results/verification_findings.md` §7 | M8, M9 (verification's own errors) |
| `results/verification_02_agreement.csv` | M8 (corrected κ SE) |
| `results/verification_03_sensitivity.csv` | M9 (corrected `n` values) |
| `src/workflow_00_model_and_prompt.md` | D1 (model summary vs changelog) |
| `data/derived/workflow_ai_predictions.csv` | D2 (stale free-tier run) |
| `results/06_transfer_annotations.txt` | D3 (timestamp decision) |
| `codebook.md` §5, §7 | D3, U1 (requirements) |
| `results/workflow_10_resume_log.txt`, `workflow_12_validate_resume.txt`, `workflow_13_validate.txt` | U6 (quota and model switch) |

*End of document.*
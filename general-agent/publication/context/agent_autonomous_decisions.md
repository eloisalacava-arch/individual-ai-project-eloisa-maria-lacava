# Agent autonomous decisions — "Can AI Really Understand Movie Reviews?"

Date: 14 November 2026  
Analyst: autonomous Agent run  
Seed: **20261114**

This file records every analytical choice made without asking for approval, the
reasoning behind it, alternatives considered, and the errors encountered while
building the analysis.

---

## 0. Scope and constraints followed

- Used the same 200 reviews and the same 200 OpenRouter predictions in
  `data/derived/workflow_13_predictions_openrouter.csv`.
- Did **not** open or read the previous Workflow analysis scripts, result
  tables, figures, or `results/workflow_40_summary.md` before completing this
  analysis. Only setup/annotation inputs, `AGENTS.md`, `plan.md`,
  `codebook.md`, and `data/README.md` were read.
- Did not modify `data/raw/`, `plan.md`, `codebook.md`, or `AGENTS.md`.
- Did not modify `data/annotations.csv`, `data/annotations.xlsx`, or
  `data/derived/sample_mapping.csv`.
- All new scripts start with `agent_`; all new outputs start with `agent_`.
- No Workflow output was overwritten.
- No new model calls were made. The 200 existing predictions were reused, so
  there is no new prompt/output/model log to write for this run. The model name
  is carried in every result table through the analysis dataset.

---

## 1. Data decisions

### D1 — Primary reference is the IMDb label
**Choice:** Treat `label` (0 = negative, 1 = positive) as the primary ground
truth, following `plan.md` and `data/README.md`.  
**Reason:** The project question is about agreement with the dataset labels; this
also keeps the result comparable with the Workflow run.  
**Alternative rejected:** Using the manual annotation as primary truth. The
manual annotation is one reader's judgment and is itself in disagreement with
the labels on some rows.  
**Consequence:** All "accuracy" headline numbers use the IMDb label.

### D2 — Manual annotation is a secondary reference, not a gold standard
**Choice:** Report AI-vs-manual agreement separately and never call it "the
true accuracy".  
**Reason:** The codebook describes a single blind annotator, not adjudication.
Using it as truth would overstate its status.  
**Consequence:** Results are presented under two references, with disagreement
between them made explicit.

### D3 — Add a three-way agreement decomposition
**Choice:** Classify every review into: all three agree; AI disagrees with both;
label/manual dispute where AI sides with label; label/manual dispute where AI
sides with manual.  
**Reason:** A binary "error" count hides whether an apparent AI error is a
label/manual dispute. This decomposition is the cleanest way to show that.  
**Alternative rejected:** Treating all 200 labels as equally reliable and
ignoring the manual annotation. That would overstate the number of genuine model
mistakes.

### D4 — No rows dropped in the main analysis
**Choice:** Keep all 200 rows.  
**Reason:** Every input file matched one-to-one, predictions were all `ok`, and
there were no missing values in the variables used. Dropping rows would reduce
an already small error count.  
**Consequence:** All main tables use n = 200. Filtering appears only in labelled
sensitivity analyses (D21).

### D5 — No duplicate handling inside the 200
**Choice:** Do not deduplicate or down-weight.  
**Reason:** All 200 sampled texts occur exactly once in the test split
(`text_n_occurrences_in_test = 1` for every row), so there are no within-sample
duplicate twins to handle.

### D6 — Text metrics computed from the cleaned blind text
**Choice:** Compute length and readability proxies from `data/derived/blind_sample.csv`,
stripping any residual `<br />` tag.  
**Reason:** Codebook §1.4 requires per-review median sentence length and median
word length, and the annotation text is the cleaned text.  
**Caveat:** Sentence splitting on `[.!?]` and newlines is a heuristic;
abbreviations can inflate sentence counts. The Flesch score uses a vowel-group
syllable heuristic, not a dictionary.

---

## 2. Outcome and category decisions

### D7 — Error = AI sentiment differs from the reference sentiment
**Choice:** Binary error indicator, no neutral/abstain option.  
**Reason:** The task and the AI output are binary.  
**Consequence:** False positives and false negatives are reported separately so
a directional bias is not hidden.

### D8 — Each flag compared with the derived `straightforward` class
**Choice:** For each of `sarcasm`, `mixed`, `ambiguous`, `complex`, compare its
error rate with reviews where all four flags are 0.  
**Reason:** Codebook §1.5 and §8 define `straightforward` as the reference and
forbid a forced-choice primary category.  
**Alternative rejected:** Assigning each review one "primary" difficulty
category. That would discard multi-label information and create arbitrary
priority rules not in the codebook.

### D9 — Report both individual flags and two composites
**Choice:** Add (a) `any_difficulty` = at least one flag, and (b)
`ambiguous_or_complex` = either of the two categories named in the plan's
hypothesis.  
**Reason:** The plan's hypothesis names ambiguity and complexity, while the
falsification condition names all four. The composites answer those two
wordings directly.  
**Consequence:** Composites are reported separately and are **not** included in
the four-flag multiple-testing correction.

### D10 — Treat categories as overlapping, not mutually exclusive
**Choice:** Analyse each binary flag separately and report pairwise overlap.  
**Reason:** The codebook defines four independent binary judgments. Mixed and
ambiguous co-occur often enough that a mutually exclusive coding would be
misleading.  
**Consequence:** The four-flag regression is explicitly exploratory.

### D11 — Ancillary analysis of manual-label disputes
**Choice:** Test whether manual-label disagreement rates differ by category.  
**Reason:** If the manual annotator disagrees with the label more often on
difficult reviews, manual-based "AI errors" are partly annotation disputes
rather than model failures. This turned out to be important for interpretation.  
**Status:** Ancillary, clearly labelled as **not** an AI-error analysis.

---

## 3. Statistical decisions

### D12 — Two-sided tests throughout
**Choice:** Two-sided Fisher exact tests, two-sided trend tests, two-sided
intervals.  
**Reason:** The hypothesis is directional, but the falsification condition is
also directional in the opposite direction. A one-sided test would make the
falsification side harder to see. Two-sided tests are the conservative default
for an exploratory error analysis.

### D13 — Wilson intervals for proportions, Newcombe for differences
**Choice:** Wilson score 95% CIs for error rates and accuracies; Newcombe
hybrid-score 95% CIs for risk differences versus the reference.  
**Reason:** With small counts (especially 0 errors), Wald intervals are poor.
Newcombe's method is designed for small independent proportions.  
**Alternative rejected:** Normal-approximation CIs (can include impossible
values and collapse at 0) and bootstrap CIs (would add Monte-Carlo noise for a
problem with an analytic small-sample solution).

### D14 — Fisher exact tests for category comparisons
**Choice:** Two-sided Fisher exact test on each 2×2 table (flag vs reference).  
**Reason:** Expected counts are small; chi-square assumptions are not met.  
**Consequence:** p-values are exact but can be conservative. In addition, every
flag-vs-reference test reuses the same 106 straightforward reviews, so the tests
are not independent of one another. They are reported as descriptive
comparisons, not as simultaneous confirmatory tests.

### D15 — Benjamini-Hochberg across the four flags only
**Choice:** Compute BH q-values separately for each outcome across the four
pre-specified flags. Composites get raw p-values only.  
**Reason:** The four flags are the codebook's pre-specified family. The
composites are summaries of the same data, not additional independent tests.  
**Consequence:** Mixed-vs-label has q just above 0.05 while its raw p is below
0.05; the report says "marginal", not "significant".

### D16 — Beta-Binomial companion analysis
**Choice:** Jeffreys-prior Beta posteriors, report P(error rate higher than
reference), 95% credible intervals, and a Monte-Carlo interval for the
difference (50,000 draws, seed 20261114).  
**Reason:** With 0–6 errors in some cells, a Bayesian summary communicates
uncertainty more intuitively than a p-value alone.  
**Alternative rejected:** A flat Beta(1,1) prior; Jeffreys is a standard
weakly informative choice and is less sensitive to the prior at small n.  
**Caveat:** The Bayesian probability is not a substitute for the frequentist
test; both are reported side by side.

### D17 — Firth penalised logistic regression for small-sample adjustment
**Choice:** Fit Firth logistic models for the composite flag, the four flags,
the length-adjusted model, and the plan-named composite. Use the inverse
Hessian of the penalised likelihood for standard errors.  
**Reason:** Only 11 label errors and 6 manual-reference errors; the complex flag
has zero label errors (separation). Ordinary MLE would be unstable or
non-finite.  
**Validation:** The same optimiser with the Firth penalty switched off was
compared with `statsmodels.Logit`; maximum absolute coefficient difference was
3.3×10⁻⁸ (`results/agent_03_firth_validation.csv`).  
**Alternative rejected:** Unpenalised logistic regression (unstable), exact
logistic regression (not available in the installed environment and harder to
extend to several flags), or dropping the four-flag model (loses the
simultaneous check).  
**Caveat:** Firth standard errors are approximate; intervals are wide by
construction and are not interpreted as precise effect sizes.

### D18 — Length adjustment using centred log2 word count
**Choice:** In the adjusted model, use `log2_words` centred at its mean.  
**Reason:** A one-unit change is a doubling of length, which is interpretable;
centring makes the intercept refer to an average-length review and improves
numerical stability.  
**Result:** The composite flag's estimate barely changes and length is not
associated with errors; this argues against length as the explanation for the
category association.

### D19 — Trend analysis by number of flags
**Choice:** Cochran-Armitage trend test on 0–4 flags, plus a collapsed 0/1/2+
version, plus Spearman correlation.  
**Reason:** The codebook treats flags as independent, so a dose-response summary
is a useful complement to the per-flag tests. The collapsed version addresses
the tiny n at 3–4 flags.  
**Caveat:** The trend test is unadjusted for flag type; it is descriptive.

### D20 — Hypothesis verdict rule fixed before interpreting the tables
**Choice:** For each category:
- point estimate of risk difference ≤ 0 → falsification condition met on the
  point estimate;
- 95% CI lower bound > 0 → hypothesis direction supported;
- otherwise → direction consistent but inconclusive.  
**Reason:** This prevents "accept the hypothesis" from being claimed on a
non-significant point estimate in the predicted direction.  
**Consequence:** Ambiguous-vs-label is called inconclusive, not supportive;
complex-vs-label is called a point-estimate falsification, not a decisive one.

### D21 — Sensitivity exclusions
**Choice:** Two sensitivity scenarios: exclude `uncertainty_sentiment = 2`
(13 rows), and exclude `uncertainty_categories = 2` (6 rows).  
**Reason:** Codebook §8 explicitly asks for sensitivity to
`uncertainty_sentiment = 2`; the second exclusion checks the flags themselves.  
**Consequence:** Cells become even smaller; results are reported as robustness
checks, not as the primary analysis. No imputation was used.

### D22 — No correction for multiple outcomes beyond the stated families
**Choice:** Do not apply a global correction across accuracy, agreement,
category, trend, complexity and sensitivity analyses.  
**Reason:** The analysis is exploratory and the families are reported
separately. A global correction would be arbitrary and would hide the
distinction between pre-specified flags and descriptive checks.  
**Caveat:** This is a limitation, stated in the report.

### D23 — Train/test overlap and duplicate caveats
**Choice:** Do not adjust accuracy for the 123 train–test overlapping texts or
the test-split duplicates.  
**Reason:** None of the 200 sampled texts is duplicated within the test split,
and the sample is the evaluation unit. The broader overlap issue is a
generalisation caveat, not something that can be corrected with 200 rows and no
model retraining. It is stated as a limitation.

---

## 4. Figure and reporting decisions

### D24 — Five figures
**Choice:**
1. Confusion matrices (AI vs label; AI vs manual).
2. Category error rates with 95% CIs, label and manual panels.
3. Error rate by number of flags.
4. Distributions of complexity proxies by human `complex` flag.
5. Three-way agreement within each category.  
**Reason:** These map directly onto the four objectives and avoid decorative
plots. Figure 5 is the key figure for the label/manual dispute interpretation.  
**Alternative rejected:** A bar chart of raw counts by category (less
informative than rates with CIs) and a scatterplot matrix of proxies (too noisy
at n=200).

### D25 — Report generated from result CSVs
**Choice:** `src/agent_06_report.py` reads the result tables and inserts the
numbers into the prose.  
**Reason:** AGENTS.md requires every number in a write-up to come from a file in
`results/`. Generating the report this way prevents transcription errors and
keeps provenance explicit.  
**Consequence:** Re-running the pipeline regenerates the report with updated
numbers.

### D26 — No causal language
**Choice:** Use "associated with", "higher error rate", "consistent with";
never "causes", "leads to", "makes the AI fail".  
**Reason:** Categories are observed, not randomised; one model, one sample.  
**Also stated:** possible pretraining exposure to IMDb cannot be ruled out from
local files, so even the accuracy estimate may be optimistic.

---

## 5. Errors encountered and corrections

### E1 — Wrong SciPy namespace in the shared utility
**What happened:** The first version of `agent_00_utils.py` called
`stats.optimize.minimize`. `scipy.stats` does not expose `optimize`.  
**How corrected:** Imported `optimize` from SciPy and changed the call to
`optimize.minimize`. Re-ran the Firth models; all converged.

### E2 — Report generator KeyError for `straightforward`
**What happened:** The compact `agent_03_category_tests.csv` does not contain a
row for the reference class; the reference values live in columns. The report
initially indexed `tl.loc['straightforward', ...]`.  
**How corrected:** Used the reference columns from the flag rows (and the
analysis dataset for reference n/errors). The report now reads only available
columns.

### E3 — Hard-coded consensus interval in the first report draft
**What happened:** The consensus accuracy CI was typed into the report
generator from the step-2 console output rather than read from a table.  
**How corrected:** Added `results/agent_02_consensus_summary.csv` to
`agent_02_accuracy.py` and made the report read the interval from it. This also
fixed the same issue for the "AI sides with manual" interval.

### E4 — Figure label overlap
**What happened:** Figure 2's category labels collided at the default width, and
Figure 3's point annotations overlapped the lines. Figure 5's legend sat on top
of the longest bar.  
**How corrected:** Widened Figure 2, shortened two-line labels, moved the
sample sizes in Figure 3 to the x-axis tick labels, and moved Figure 5's legend
below the plot. Re-rendered and visually checked all five figures.

### E5 — Conceptual mix-up between "manual error" and "manual-label dispute"
**What happened:** During early exploration I briefly conflated the 15
manual-versus-label disagreements with the 6 AI-versus-manual errors. They are
different quantities.  
**How corrected:** The scripts compute `error_vs_manual` and
`manual_disagrees_with_label` as separate variables, and the report uses the
three-way decomposition. The distinction is now central to the interpretation.

### E6 — Missing annotation timestamps
**What happened:** `annotation_timestamp` is missing for all 200 rows in the CSV
(likely lost in the spreadsheet export).  
**How handled:** The field is not used in any analysis. This is reported in the
prepare log, not treated as a data-quality failure for the sentiment variables.

### E7 — No intra-annotator reliability data
**What happened:** Codebook §7 asks for a second-pass reliability check on about
20 reviews. The provided annotation workbook has no second-pass sheet.  
**How handled:** No reliability estimate is fabricated. The limitation is stated
in the report. I am not sure how much of the label/manual disagreement is
annotator error.

---

## 6. What I deliberately did not do

- Did not re-run the AI model. The task required the existing 200 predictions
  for comparability, and re-running would introduce a new prompt/version
  confound.
- Did not train a classifier on the 200 rows. With 11 errors and four
  overlapping flags, a trained model would overfit and would not answer the
  descriptive question.
- Did not impute missing values. There were none in the analysed variables.
- Did not remove outliers by review length. Length was tested as a proxy and
  was not associated with errors; trimming would be an arbitrary choice.
- Did not collapse the four flags into a single severity score for the primary
  test. The composite `any_difficulty` is reported alongside, but the per-flag
  tests preserve the codebook's definitions.
- Did not claim that the manual annotation is more accurate than the IMDb
  label. The McNemar test gives no evidence for that, and there is no
  adjudicated third reference.
- Did not claim that the model "understands" or "does not understand" reviews.
  The data support statements about classification agreement and associations
  with review characteristics, not about understanding.

---

## 7. Bottom line of the autonomous choices

The choices above make the analysis **conservative about error counts**
(separating label/manual disputes from AI errors), **conservative about
significance** (exact tests, BH correction, wide small-sample intervals), and
**explicit about uncertainty** (two references, Bayesian companion, sensitivity
exclusions). The main substantive conclusion is that the plan's hypothesis is
not cleanly supported: ambiguity is directionally harder but inconclusive,
complexity is not harder against IMDb labels, and the strongest association is
with mixed sentiment. Most apparent AI errors are rows where the two human
references disagree, which makes the label-based error rate an upper bound on
genuine model error rather than a clean estimate of it.
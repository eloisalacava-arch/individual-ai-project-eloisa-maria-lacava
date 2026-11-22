# Independent Verification — IMDb Sentiment Error Analysis

**Prepared:** 2026-11-16 · **Verifier:** independent re-analysis (this session)
**Rule followed (AGENTS.md):** every number below comes from a file in
`results/`; the generating script is named for each.

---

## 0. Scope, independence, and inputs

I re-analysed the project from scratch. Before comparing anything, I did **not**
open any `workflow_*` or `agent_*` analysis script, result, figure, or report. I
wrote my own code (`src/verification_*.py`) using only numpy/pandas/scipy/
statsmodels, imported nothing from the project's earlier code, and saved every
independent number under `results/verification_*`. Only after that did I read the
prior analyses to compare (§6).

| Item | Value |
|---|---|
| Analysis file | `data/derived/workflow_joined.csv`, 200 rows |
| Model under test | `openrouter/deepseek/deepseek-v4.1-flash-1017` |
| Manual reference | single blind annotator (<annotator>), codebook flags |
| Primary outcome | AI prediction vs IMDb label ("error" = disagree) |
| Seeds | kappa/delta checks seeded 20261116 (`verification_02`, `_utils`) |

Scripts and outputs (see full manifest in §8):

- `src/verification_01_data_audit.py` → `results/verification_01_audit.txt`
- `src/verification_02_accuracy.py` → `results/verification_02_*.csv/_log.txt`
- `src/verification_03_categories.py` → `results/verification_03_*.csv/_log.txt`
- `src/verification_04_hypothesis.py` → `results/verification_04_*`
- `src/verification_05_figures.py` → `results/figures/verification_fig*.png`
- `src/verification_06_compare.py` → `results/verification_06_*`
- shared helpers: `src/verification_utils.py`

**Headline:** my independent numbers reproduce the prior results for every
shared quantity. The prior conclusions hold. I did not find any calculation error
that changes a reported result. I did find several documentation/provenance
gaps and two bugs in **my own** first draft, both fixed (§7).

---

## 1. Data audit (Q: is the 200-row file trustworthy?)

Source: `results/verification_01_audit.txt`.

- 200 rows, 200 unique `review_id`, 200 unique texts, 0 fully duplicated rows,
  no nulls in required fields.
- `human_straightforward` is derived correctly (0/200 mismatches with
  "all four flags = 0").
- `imdb_label` and `imdb_label_text` are internally consistent.
- **Independent label cross-check:** after reproducing the codebook's cleaning
  (`<br />` → newline), **200/200** joined texts are found in the raw test split,
  **0 label mismatches**, and **200/200** `test_row_index` values point at the
  matching text. (A naive raw-text join matches only 82/200, which is expected —
  the cleaning step must be applied first; this confirms the cleaning was
  applied, not that data are wrong.)
- **Annotation cross-check:** all human columns match `data/annotations.csv`
  exactly (0 discrepancies on sentiment, all four flags, straightforward, both
  uncertainty fields, annotator).
- Flag counts: sarcasm 25, mixed 50, ambiguous 40, complex 20, straightforward
  106, any-flag 94. `text_n_occurrences_in_test` = 1 for all 200 sampled texts.

**Documentation issues found (no impact on the numbers):**

1. `annotation_timestamp` is blank in **200/200** rows, although codebook §5
   requires an ISO timestamp per row. The audit and the join log both record
   this; `data/annotations.csv` was not modified.
2. No intra-annotator reliability sheet exists, though codebook §7 requires
   re-annotating ~20 reviews; this is stated as a limitation in the prior
   reports (`results/workflow_40_summary.md` §9.1, `agent_report.md`).
3. `data/derived/workflow_ai_predictions.csv` holds the **stale free-tier run**
   (model `opencode/deepseek-v4.1-flash-free`) and disagrees with the analysed
   predictions on **129/200** rows. The analysed predictions are in
   `data/derived/workflow_13_predictions_openrouter.csv` and match the joined
   file **200/200**. The stale file is documented as unused
   (`results/workflow_40_summary.md` §3), but its generic name is a hazard.

---

## 2. Q1 — Overall AI accuracy against IMDb labels

Source: `results/verification_02_metrics.csv`, `verification_02_confusions.csv`.

**Accuracy = 189/200 = 0.9450** (Wilson 95% CI 0.9042–0.9690).

| | IMDb positive | IMDb negative |
|---|---|---|
| **AI positive** | 98 (TP) | 7 (FP) |
| **AI negative** | 4 (FN) | 91 (TN) |

- Sensitivity (positive) 0.9608; specificity (negative) 0.9286.
- Balanced accuracy 0.9447; macro-F1 0.9449; MCC 0.8903; κ(AI, IMDb) 0.8899.
- Reference: **manual vs IMDb = 185/200 = 0.9250** (0.8800–0.9540). The two
  human references disagree on 15/200 reviews, so "error" depends on which
  reference is used.

---

## 3. Q2 — AI agreement with manual annotations

Sources: `verification_02_metrics.csv`, `verification_02_agreement.csv`,
`verification_02_threeway.csv`, `verification_02_mcnemar_ai_vs_manual.csv`.

- **Agreement = 194/200 = 0.9700** (Wilson 95% CI 0.9361–0.9862).
- **Cohen's κ = 0.9399** (multinomial-delta SE 0.0242; percentile bootstrap 95%
  CI 0.8895–0.9800, 20 000 draws, seed 20261116). "Almost perfect".
- Flow of the 200 cells against the same IMDb labels: both correct 184, both
  wrong 10, AI-right/manual-wrong 5, AI-wrong/manual-right 1. Exact McNemar
  (6 discordant) p = 0.2188 — no evidence the two predictions differ in
  accuracy against IMDb.
- Three-way: all three agree 184/200 (0.920); AI = manual but both ≠ IMDb 10/200.

**Important caveat reproduced independently:** of the 6 AI-vs-manual
disagreements, **5** are rows where the manual annotation disagrees with the
IMDb label and the AI sides with the IMDb label, and 1 is a row where AI
disagrees with both (`verification_06_manual_error_decomposition.csv`). Manual-
referenced "errors" therefore mostly reflect label/manual disputes, not proof
that the AI misread the text.

---

## 4. Q3 — Error rates by review category (vs IMDb)

Source: `results/verification_03_category_rates.csv`,
`verification_03_category_tests.csv`.

| Category | n | Errors | Error rate | Wilson 95% CI | vs manual |
|---|---|---|---|---|---|
| sarcasm | 25 | 2 | 0.0800 | 0.0222–0.2497 | 0.0400 |
| mixed | 50 | 6 | 0.1200 | 0.0562–0.2380 | 0.0800 |
| ambiguous | 40 | 3 | 0.0750 | 0.0258–0.1986 | 0.1250 |
| complex | 20 | 0 | 0.0000 | 0.0000–0.1611 | 0.1000 |
| **straightforward** | **106** | **2** | **0.0189** | 0.0052–0.0662 | 0.0000 |
| any flag | 94 | 9 | 0.0957 | 0.0512–0.1720 | 0.0638 |

Each category vs the disjoint straightforward reference (positive RD = higher
error = lower accuracy):

| Category | Risk diff. | Newcombe 95% CI | Fisher p | Holm p |
|---|---|---|---|---|
| sarcasm | +0.0611 | −0.0136 to +0.2313 | 0.1644 | 0.3799 |
| **mixed** | **+0.1011** | **+0.0217 to +0.2200** | **0.0137** | **0.0549** |
| ambiguous | +0.0561 | −0.0121 to +0.1805 | 0.1266 | 0.3799 |
| complex | −0.0189 | −0.0662 to +0.1428 | 1.0000 | 1.0000 |
| any flag | +0.0769 | +0.0119 to +0.1544 | 0.0264 | — |

- Only **mixed** has a CI excluding zero, and it is borderline after correction
  (Holm 0.055).
- **Overlap check** (`verification_06_isolated_rates.csv`): isolated vs
  co-occurring error rates — sarcasm 1/17 vs 1/8; **mixed 4/24 vs 2/26**;
  **ambiguous 2/13 vs 1/27**; complex 0/6 vs 0/14. Ambiguous and mixed look
  harder when isolated, but these cells are tiny.
- **Trend** across number of flags (`verification_03_trend.csv`): 0 flags
  2/106 (1.9%), 1 flag 7/60 (11.7%), 2 flags 2/28 (7.1%), 3 flags 0/5, 4 flags
  0/1. Cochran–Armitage z = 1.3165, p = 0.1880 — no significant monotone trend.
- **Multivariable Firth logistic** (`verification_03_logistic.csv`, error ~ four
  flags): mixed coef +1.417 (p = 0.022) is the only term that survives;
  sarcasm +0.730 (p = 0.335), ambiguous +0.278 (p = 0.688), complex −1.403
  (p = 0.338). Standard MLE does not converge (complex is completely separated:
  0/20 errors).
- **Sensitivity, excluding the 13 `uncertainty_sentiment` = 2 rows**
  (`verification_03_sensitivity.csv`): overall error 8/187 = 0.0428; by category
  sarcasm 2/23 = 0.087, mixed 5/44 = 0.114, ambiguous 0/27 = 0.000, complex
  0/15 = 0.000, straightforward 2/106 = 0.019. No conclusion reverses.

---

## 5. Q4 — Is the original hypothesis supported or falsified?

**Hypothesis (plan.md §3):** accuracy is lower for **ambiguous or linguistically
complex** reviews than for straightforward reviews.
**Falsification (plan.md §4):** challenged if sarcastic/mixed/ambiguous/complex
reviews perform **as well or better**.

Sources: `results/verification_04_hypothesis_summary.csv`, `_mde.csv`.

| Group | Accuracy | RD(errors) vs straight | 95% CI | Fisher p | Verdict |
|---|---|---|---|---|---|
| ambiguous | 0.925 | +0.056 | −0.012 to +0.181 | 0.127 | direction consistent, **inconclusive** |
| complex | **1.000** | −0.019 | −0.066 to +0.143 | 1.000 | point est. **opposite** direction |
| ambiguous OR complex (plan's named union) | 0.940 | +0.041 | −0.020 to +0.144 | 0.328 | **inconclusive** |
| mixed | 0.880 | +0.101 | +0.022 to +0.220 | 0.014 | lower accuracy |
| sarcasm | 0.920 | +0.061 | −0.014 to +0.231 | 0.164 | inconclusive |

**Verdict: the hypothesis as stated is NOT supported by these data.** Neither
category named in the hypothesis shows a statistically distinguishable accuracy
drop; **complex** reviews had the *highest* point accuracy (0/20 errors), so the
falsification condition is met on the point estimate for that category. The one
robust difficulty signal is **mixed opinions**, which the hypothesis did not
name. I cannot call the hypothesis *false* with confidence: the intervals are
wide and compatible with a real moderate-sized decrement for ambiguous reviews.

**Power is the key limit** (`verification_04_mde.csv`): with only 11 total AI
errors, the minimum detectable error-rate difference vs straightforward at 80%
power is ~21 pp for ambiguous, ~29 pp for complex, ~19 pp for mixed, ~15 pp for
any-flag. Small-to-moderate effects are simply not detectable here.

---

## 6. Comparison with the previous analyses

Source: `results/verification_06_comparison.csv`, `_log.txt`.

Every shared quantity matches the prior `workflow_30` and `agent` outputs to the
reported precision: accuracy 0.9450, agreement 0.9700, κ 0.9399, all four
category error rates, all risk differences and Fisher p-values, the per-category
sensitivity rates, and the Firth coefficients (identical to 3 d.p.). My
conclusions match both prior reports: hypothesis not supported as stated, mixed
opinions the one robust signal.

**Methodological differences (none changes a conclusion):**

| # | Topic | My verification | Prior | Effect |
|---|---|---|---|---|
| a | Multiple testing | Holm–Bonferroni | workflow_30: none; agent_03: BH q | Only mixed is nominally significant either way (Holm 0.055 / BH 0.055) |
| b | Adjusted logistic | Firth/Jeffreys | workflow_30: ridge 1e-4; agent_03: Firth | ridge cannot estimate complex (separation); Firth returns a finite, non-significant coef |
| c | Trend variance | N convention (z 1.3165, p 0.188) | agent N−1 (z 1.3132, p 0.189) | immaterial |
| d | κ standard error | delta method 0.0242 | agent closed-form 0.0245 | immaterial (bootstrap agrees) |

---

## 7. Errors and discrepancies discovered

**In my own verification (found and fixed during this task):**

1. **Kappa SE was inflated.** My first closed-form (Fleiss) kappa SE gave 0.0334;
   bootstrap and a Monte-Carlo simulation both gave ≈0.0242. I replaced it with a
   numerical multinomial-delta method validated against the bootstrap
   (`src/verification_utils.py::kappa_se`). Final SE 0.0242.
2. **Per-category sensitivity mis-labelled.** In my first `verification_03`
   draft, a reused `keep` variable made the "exclude uncertainty_sentiment = 2"
   per-category rows actually use the stricter "exclude any uncertainty" filter
   (e.g. ambiguous n = 6 instead of 27). Fixed with a dedicated `keep_sent`
   variable; the corrected output now matches the prior sensitivity table
   exactly.

**In the prior work (documented already; no result changes):**

3. `annotation_timestamp` blank in all 200 rows (codebook §5 requirement unmet;
   recorded as a source limitation).
4. Planned intra-annotator reliability re-annotation (codebook §7) not
   performed; both prior reports state this.
5. `data/derived/workflow_ai_predictions.csv` is the stale free-tier run (129/200
   predictions differ from the analysed run); documented as unused, but the file
   name does not signal that.
6. `src/workflow_00_model_and_prompt.md` §1 still lists the **free-tier** model
   as primary; the executed/analysed model is the OpenRouter one, recorded only
   in the v1.3 changelog of the same file. Not an error, but the top of the
   document is stale relative to the run.
7. The `complex` category previously produced a meaningless ridge odds ratio
   under complete separation; `workflow_40_summary.md` §10 already documents the
   detection and correction (reported as not estimable). My Firth analysis agrees
   that the estimate is driven by the 0/20 cell and is not interpretable.

**Interpretation cautions I independently confirm:**

8. The manual reference is not an adjudicated gold standard; 15/200 manual labels
   disagree with IMDb, and the manual-referenced category "errors" are mostly
   those disputes (5 of 6).
9. The difficulty flags and the manual sentiment come from the same annotator, so
   manual-referenced category comparisons are partly circular.

---

## 8. Limitations and honest uncertainty

- **Small number of errors (11) → low power.** Every category test above is
  underpowered; "indistinguishable" is not "equivalent". I am not sure whether a
  true moderate association exists for ambiguous or complex reviews.
- Single annotator, no inter- or intra-annotator reliability; the category flags
  inherit any annotation error. I am not sure how much of the residual error is
  AI error versus label/noise error.
- IMDb labels come from star ratings, not from reading the text, so the primary
  "truth" is itself imperfect.
- Multi-label categories overlap; category-vs-straightforward comparisons share
  the same reference group, so the four p-values are dependent (descriptive, not
  simultaneous confirmatory tests).
- No causal claims: all comparisons are observational associations in one
  200-review convenience sample.
- The model's temperature was not controllable and the serving endpoint is not
  recorded, so run-to-run stability is unmeasured.

**Bottom line I can defend:** on this sample the AI is ~94.5% accurate against
IMDb labels and agrees with the human annotator on 97% of reviews; the stated
hypothesis about ambiguity/complexity is not supported; the only category with a
demonstrable (borderline) accuracy drop is **mixed opinions**, and even that is
too small a sample to be conclusive.

---

## 9. File manifest (all `verification_`-prefixed; data files untouched)

| File | Contents |
|---|---|
| `results/verification_01_audit.txt` | data audit + raw-label and annotation cross-checks |
| `results/verification_02_metrics.csv` | accuracy/precision/recall/F1/MCC/kappa, overall and by class |
| `results/verification_02_confusions.csv` | three confusion matrices |
| `results/verification_02_agreement.csv` | AI–manual agreement, kappa, SE, bootstrap CI |
| `results/verification_02_mcnemar_ai_vs_manual.csv` | paired accuracy test |
| `results/verification_02_threeway.csv` | three-way agreement |
| `results/verification_03_category_rates.csv` | error rates by category (vs IMDb and vs manual) |
| `results/verification_03_category_tests.csv` | RD, Newcombe CI, Fisher p, Holm p vs straightforward |
| `results/verification_03_trend.csv`, `_trend_test.csv` | error by number of flags; Cochran–Armitage |
| `results/verification_03_logistic.csv` | standard + Firth logistic (four flags) |
| `results/verification_03_sensitivity.csv` | overall and per-category sensitivity |
| `results/verification_03_rates_vs_manual.csv` | error rates using the manual reference |
| `results/verification_04_hypothesis_summary.csv` | hypothesis/falsification verdict table |
| `results/verification_04_mde.csv` | minimum detectable effects (power) |
| `results/verification_06_comparison.csv` | side-by-side vs prior analyses |
| `results/verification_06_manual_error_decomposition.csv` | AI–manual error decomposition |
| `results/verification_06_isolated_rates.csv` | isolated vs co-occurring category rates |
| `results/verification_02_log.txt`, `_03_log.txt`, `_04_log.txt`, `_06_log.txt` | full logs |
| `results/figures/verification_fig1..4_*.png` | error rates, confusion, trend, risk differences |

*No causal language is intended. Prepared independently on 2026-11-16.*

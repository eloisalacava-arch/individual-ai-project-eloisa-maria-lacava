# AI versus human sentiment classification - IMDb case study — study report

| Field | Value |
|---|---|
| Run id | `imdb_case_study` |
| Generated | 2026-11-20T11:33:26.725103+00:00 |
| Model | `openrouter/deepseek/deepseek-v4.1-flash-1017` |
| Seed | 20261116 |
| Analysed rows | 200 |
| Categories | positive / negative |
| Qualitative analysis | included (hypotheses only) |
| Reference | Primary: manual annotation of 200 reviews. Secondary: star-derived IMDb label (0/1). |

#### Executive summary

- 200 observations were classified into 2 categories against the human reference.
- Agreement: accuracy 0.970 (95% CI 0.936–0.986), balanced accuracy 0.970, Cohen's kappa 0.940 (95% CI 0.889–0.980).
- 6 disagreements; the majority class `positive` is 51.5% of the sample.
- Weakest category: `negative` (recall 0.959 on 97 reference rows).
- Qualitative hypotheses exist for 16 sampled items; the hypothesis most concentrated in disagreements is ambiguous reference label (83.3% above the control).
- The plan's hypothesis is tested against the star-derived IMDb proxy label. Verdict on the pre-specified hypothesis (`ambiguous_or_complex`): against the star-derived IMDb proxy label the hypothesis is lower accuracy in the hypothesised direction, but inconclusive (the interval includes zero) (risk difference +4.1 pp, 95% CI -2.0 to +14.4, Fisher p = 0.328); against the manual human annotation the hypothesis is significantly lower accuracy for the subgroup, supporting the hypothesis (risk difference +12.0 pp, 95% CI +4.7 to +23.8, Fisher p = 0.001).

> Explanations are AI-generated hypotheses, not confirmed causes. The reference is a benchmark, not adjudicated truth, and no causal claim is made.

## 1. Study description

- Task: a sentiment classifier for movie reviews.
- Decision rule: Judge the reviewer's opinion of the film itself, not of other reviewers, the film industry, or anything else. If the review is mixed or ironic, follow the reviewer's final overall verdict about the film.
- Reference: Primary: manual annotation of 200 reviews. Secondary: star-derived IMDb label (0/1).
- Why this primary reference: The 200 manual annotations are the primary reference: they are a blind, text-based reading of each review, so they share the same construct as the classifier's task. The star-derived IMDb label is not derived from the review text. Both are reported; neither is adjudicated truth.
- Categories and definitions:
  - `positive`: The reviewer's overall stance toward the film is favourable.
  - `negative`: The reviewer's overall stance toward the film is unfavourable.
- Ordered categories: False.
- Sampling: seeded, unstratified (seed 20261116, n=None, unique_text=True).
- Predictions were frozen before the reference was opened (stage 05); the scored copy is `results/frozen_predictions.csv`.

## 2. Agreement with the reference

| Metric | Value | 95% CI | Method |
|---|---|---|---|
| Accuracy | 0.970 | 0.936–0.986 | Wilson |
| Balanced accuracy | 0.970 | 0.944–0.990 | bootstrap |
| Cohen's kappa (unweighted) | 0.940 | 0.889–0.980 | bootstrap |
| Cohen's kappa (quadratic) | not requested |  | bootstrap |

Accuracy counts exact matches; kappa corrects for chance and depends on the marginal distribution; balanced accuracy averages per-class recall.

Source: `results/t02_accuracy.csv, t03_kappa.csv`

## 3. Confusion matrix

| Human label \ AI answer | positive | negative | total_human |
|---|---|---|---|
| positive | 101 | 2 | 103 |
| negative | 4 | 93 | 97 |
| total_predicted | 105 | 95 | 200 |

Rows are the human label; columns are the AI answer. The final row and column are marginals.

Source: `results/t04_confusion_matrix.csv`

<!-- figure: confusion_matrix | Confusion matrix, rows = human, columns = AI -->

## 4. Per-category results

| Class | Reference n | Predicted n | Precision | Recall (95% CI) | F1 (95% CI) |
|---|---|---|---|---|---|
| positive | 103 | 105 | 0.962 | 0.981 (0.932–0.995) | 0.971 (0.946–0.991) |
| negative | 97 | 95 | 0.979 | 0.959 (0.899–0.984) | 0.969 (0.941–0.990) |

Source: `results/t05_per_class_metrics.csv`

<!-- figure: per_class_recall | Per-category recall with 95% confidence interval -->

## 5. Class distribution

| Class | Human n | Human share | AI n | AI share | Difference (AI minus human) |
|---|---|---|---|---|---|
| positive | 103 | 51.5% | 105 | 52.5% | 1.0% |
| negative | 97 | 48.5% | 95 | 47.5% | -1.0% |

Source: `results/t06_class_distribution.csv`

## 6. Disagreement structure

Total disagreements: **6**.

| Bucket | n | Share |
|---|---|---|
| Agree | 194 | 97.0% |
| Opposite | 6 | 3.0% |

The `Share` column is each bucket's share of all 200 analysed rows, not of the 6 disagreements; the buckets sum to 100%.

With two categories every disagreement is at maximum distance, so `Adjacent` is always zero.

Most frequent disagreement directions:

| Reference | Prediction | n | Share of disagreements |
|---|---|---|---|
| negative | positive | 4 | 66.7% |
| positive | negative | 2 | 33.3% |

Source: `results/t07_ordinal_errors.csv, t11_disagreement_pairs.csv`

## 7. Subgroup hypothesis test

Pre-specified hypothesis: AI classification accuracy will be lower for ambiguous or linguistically complex reviews than for straightforward reviews.

Risk difference is the subgroup error rate minus the reference rate, so a positive value means lower accuracy for the subgroup. The interval is a Newcombe hybrid score; q is Benjamini-Hochberg across the pre-specified flags only, so composite rows carry no q. This is an observational association on one sample, not a causal effect.

| Reference | Subgroup | Family | Subgroup n/err | Reference n/err | Rate (subgroup) | Rate (reference) | Risk diff (pp) | 95% CI (pp) | Fisher p | BH q | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| manual human annotation | human_sarcasm | pre_specified | 25/1 | 106/0 | 4.0% | 0.0% | +4.0 | -0.8 to +19.5 | 0.191 | 0.191 | lower accuracy in the hypothesised direction, but inconclusive (the interval includes zero) |
| manual human annotation | human_mixed | pre_specified | 50/4 | 106/0 | 8.0% | 0.0% | +8.0 | +2.0 to +18.8 | 0.010 | 0.019 | significantly lower accuracy for the subgroup, supporting the hypothesis |
| manual human annotation | human_ambiguous | pre_specified | 40/5 | 106/0 | 12.5% | 0.0% | +12.5 | +4.6 to +26.1 | 0.001 | 0.005 | significantly lower accuracy for the subgroup, supporting the hypothesis |
| manual human annotation | human_complex | pre_specified | 20/2 | 106/0 | 10.0% | 0.0% | +10.0 | +2.0 to +30.1 | 0.024 | 0.032 | significantly lower accuracy for the subgroup, supporting the hypothesis |
| manual human annotation | ambiguous_or_complex | composite | 50/6 | 106/0 | 12.0% | 0.0% | +12.0 | +4.7 to +23.8 | 0.001 | — | significantly lower accuracy for the subgroup, supporting the hypothesis |
| manual human annotation | any_difficulty | composite | 94/6 | 106/0 | 6.4% | 0.0% | +6.4 | +1.5 to +13.2 | 0.010 | — | significantly lower accuracy for the subgroup, supporting the hypothesis |
| star-derived IMDb proxy label | human_sarcasm | pre_specified | 25/2 | 106/2 | 8.0% | 1.9% | +6.1 | -1.4 to +23.1 | 0.164 | 0.219 | lower accuracy in the hypothesised direction, but inconclusive (the interval includes zero) |
| star-derived IMDb proxy label | human_mixed | pre_specified | 50/6 | 106/2 | 12.0% | 1.9% | +10.1 | +2.2 to +22.0 | 0.014 | 0.055 | lower accuracy for the subgroup with an interval excluding zero, but not significant after Benjamini-Hochberg correction (q = 0.055) |
| star-derived IMDb proxy label | human_ambiguous | pre_specified | 40/3 | 106/2 | 7.5% | 1.9% | +5.6 | -1.2 to +18.1 | 0.127 | 0.219 | lower accuracy in the hypothesised direction, but inconclusive (the interval includes zero) |
| star-derived IMDb proxy label | human_complex | pre_specified | 20/0 | 106/2 | 0.0% | 1.9% | -1.9 | -6.6 to +14.3 | 1.000 | 1.000 | higher accuracy, against the hypothesised direction, but inconclusive (the interval includes zero) |
| star-derived IMDb proxy label | ambiguous_or_complex | composite | 50/3 | 106/2 | 6.0% | 1.9% | +4.1 | -2.0 to +14.4 | 0.328 | — | lower accuracy in the hypothesised direction, but inconclusive (the interval includes zero) |
| star-derived IMDb proxy label | any_difficulty | composite | 94/9 | 106/2 | 9.6% | 1.9% | +7.7 | +1.2 to +15.4 | 0.026 | — | significantly lower accuracy for the subgroup, supporting the hypothesis |

Source: `results/t14_subgroup_tests.csv`

### Do manual human annotation and star-derived IMDb proxy label disagree more on difficult reviews?

A dispute is a row where manual human annotation and star-derived IMDb proxy label carry different labels. Where the dispute rate is higher than the straightforward reference, the two references are less settled, so a model disagreement there is weaker evidence of a model error.

| Comparison | Subgroup | Subgroup n/disputes | Reference n/disputes | Rate (subgroup) | Rate (reference) | Risk diff (pp) | 95% CI (pp) | Fisher p | BH q |
|---|---|---|---|---|---|---|---|---|---|
| human_sentiment_vs_imdb_label | human_sarcasm | 25/3 | 106/2 | 12.0% | 1.9% | +10.1 | +1.0 to +28.1 | 0.048 | 0.064 |
| human_sentiment_vs_imdb_label | human_mixed | 50/8 | 106/2 | 16.0% | 1.9% | +14.1 | +5.1 to +26.7 | 0.002 | 0.007 |
| human_sentiment_vs_imdb_label | human_ambiguous | 40/6 | 106/2 | 15.0% | 1.9% | +13.1 | +3.9 to +27.3 | 0.006 | 0.011 |
| human_sentiment_vs_imdb_label | human_complex | 20/2 | 106/2 | 10.0% | 1.9% | +8.1 | -0.5 to +28.3 | 0.118 | 0.118 |
| human_sentiment_vs_imdb_label | ambiguous_or_complex | 50/7 | 106/2 | 14.0% | 1.9% | +12.1 | +3.6 to +24.4 | 0.005 | — |
| human_sentiment_vs_imdb_label | any_difficulty | 94/13 | 106/2 | 13.8% | 1.9% | +11.9 | +4.6 to +20.5 | 0.002 | — |

Source: `results/t16_reference_dispute.csv`

## 8. Three-way agreement

This table decomposes every row by whether the AI prediction matches the primary reference (manual human annotation) and the secondary reference (star-derived IMDb proxy label). The `manual human annotation and star-derived IMDb proxy label agree; AI differs` row is the only kind where the two references agree and the AI differs; on the `references differ` rows the two references already disagree, so a disagreement there is not established as a model error. A secondary reference may be a proxy label rather than a second human annotation.

| Comparison | Category | n | Share | 95% CI |
|---|---|---|---|---|
| manual human annotation vs star-derived IMDb proxy label | All three agree | 184 | 92.0% | 87.4%-95.0% |
| manual human annotation vs star-derived IMDb proxy label | manual human annotation and star-derived IMDb proxy label agree; AI differs | 1 | 0.5% | 0.1%-2.8% |
| manual human annotation vs star-derived IMDb proxy label | References differ; AI agrees with manual human annotation | 10 | 5.0% | 2.7%-9.0% |
| manual human annotation vs star-derived IMDb proxy label | References differ; AI agrees with star-derived IMDb proxy label | 5 | 2.5% | 1.1%-5.7% |
| manual human annotation vs star-derived IMDb proxy label | References differ; AI agrees with neither | 0 | 0.0% | 0.0%-1.9% |

Source: `results/t15_three_way_agreement.csv`

## 9. Observation length (exploratory)

| Group | Median words |
|---|---|
| Agreement | 161.5 |
| Disagreement | 204.0 |

Seeded permutation test p = 0.359; one pre-specified feature, no multiplicity adjustment (exploratory).

Source: `results/t12_text_profile.csv, t12b_length_test.csv`

## 10. Secondary reference structure

| Comparison | n | Agreement | Kappa |
|---|---|---|---|
| ai_vs_imdb_label | 200 | 0.945 | 0.890 |
| human_sentiment_vs_imdb_label | 200 | 0.925 | 0.850 |

Secondary-reference rows describe how references relate to each other and to the model; they do not confirm a model error.

Source: `results/t13_reference_agreement.csv`

## 11. Qualitative findings

Explanations below are AI-generated hypotheses, not confirmed causes, and not proof that the classifier or the human made a mistake.

Selection coverage: 6 of 6 disagreements and 10 of 194 agreements were sampled for explanation.

| Group | Selected | Explained | Failed | Explained share |
|---|---|---|---|---|
| Disagreement | 6 | 6 | 0 | 100.0% |
| Agreement | 10 | 10 | 0 | 100.0% |

No multiplicity adjustment is applied to these reason contrasts; they are exploratory hypotheses, and the control sample is small.

A reply can carry up to three reasons, so shares do not sum to 100%.

### Disagreements

| Reason | n | Share of explained | Share of selected | Top directions |
|---|---|---|---|---|
| Mixed or conflicting signals | 6 | 100.0% | 100.0% |  (6) |
| Ambiguous reference label | 5 | 83.3% | 83.3% |  (5) |
| Subtle or low-intensity sentiment | 3 | 50.0% | 50.0% |  (3) |
| Overlapping category definitions | 2 | 33.3% | 33.3% |  (2) |
| Figurative or sarcastic language | 1 | 16.7% | 16.7% |  (1) |

### Control (agreements)

| Reason | n | Share of explained | Share of selected | Top directions |
|---|---|---|---|---|
| Mixed or conflicting signals | 7 | 70.0% | 70.0% |  (7) |
| Needs domain or world knowledge | 5 | 50.0% | 50.0% |  (5) |
| Subtle or low-intensity sentiment | 4 | 40.0% | 40.0% |  (4) |
| Figurative or sarcastic language | 1 | 10.0% | 10.0% |  (1) |
| Other | 1 | 10.0% | 10.0% |  (1) |
| Text noise or truncation | 1 | 10.0% | 10.0% |  (1) |

### Contrast with the control sample

| Reason | Disagreements | Control | Difference (disagreements minus control) |
|---|---|---|---|
| Ambiguous reference label | 83.3% | 0.0% | 83.3% |
| Overlapping category definitions | 33.3% | 0.0% | 33.3% |
| Mixed or conflicting signals | 100.0% | 70.0% | 30.0% |
| Subtle or low-intensity sentiment | 50.0% | 40.0% | 10.0% |
| Figurative or sarcastic language | 16.7% | 10.0% | 6.7% |
| Other | 0.0% | 10.0% | -10.0% |
| Text noise or truncation | 0.0% | 10.0% | -10.0% |
| Needs domain or world knowledge | 0.0% | 50.0% | -50.0% |

A reason common to both groups is not specific to disagreements. The control sample is small and its own explanations are hypotheses.

### Illustrative examples

Illustrative examples are omitted from this public copy to avoid redistributing source text. The original examples remain in the author's working tree and are not part of this repository.

Source for the aggregate qualitative tables: `results/qualitative/tables/q02_patterns.csv`, `q05_coverage.csv`, `q07_reason_contrast.csv`.

## 12. Interpretation

- The model agreed with the human reference on 0.970 of 200 analysed rows (kappa 0.940); balanced accuracy 0.970 shows this is not purely a dominant-class effect.
- The majority reference class `positive` is 51.5% of the sample, which raises chance agreement and makes kappa harder to compare across studies.
- Recall ranged from 0.959 for `negative` to 0.981 for `positive`; the lower figure means that category's reference rows were matched less often. It does not say how many rows that is, or why.
- The AI predicted `positive` 1.0 percentage points more often than the human reference did.
- With two categories every disagreement is at maximum distance, so severity cannot be graded.
- Qualitatively, ambiguous reference label was the hypothesis most concentrated in disagreements (83.3% above the control).
- Other appeared in both disagreements and the control, so it is not specific to disagreement.

## 13. Conclusions

- On this sample the model agreed with the reference on 0.970 of rows, with kappa 0.940.
- The category most often missed was `negative` (recall 0.959).
- Qualitatively, ambiguous reference label was the hypothesis most concentrated in disagreements; it, and every other hypothesis here, remains unconfirmed.
- The plan's hypothesis is tested against the star-derived IMDb proxy label. Verdict on the pre-specified hypothesis (`ambiguous_or_complex`): against the star-derived IMDb proxy label the hypothesis is lower accuracy in the hypothesised direction, but inconclusive (the interval includes zero) (risk difference +4.1 pp, 95% CI -2.0 to +14.4, Fisher p = 0.328); against the manual human annotation the hypothesis is significantly lower accuracy for the subgroup, supporting the hypothesis (risk difference +12.0 pp, 95% CI +4.7 to +23.8, Fisher p = 0.001).
- These are descriptive associations within one sample; no causal claim is made, and a disagreement is not a confirmed model error.

## 14. Limitations

- The reference is a benchmark label, not an adjudicated gold standard; a disagreement is not a confirmed model error.
- Accuracy and recall use Wilson intervals; kappa, balanced accuracy, precision and F1 use a seeded percentile bootstrap.
- All intervals are conditional on this sample; the sample is not stratified.
- No causal claim is made. Statements are descriptive associations within one sample.
- Temperature is not controlled and model versions drift.
- The 200 predictions were produced by a previous run and are reused verbatim; this stage makes no model calls.
- The star-derived IMDb label comes from ratings, not from reading the text.
- Pretraining contamination with IMDb-like text cannot be ruled out.
- No human-human agreement estimate exists for the manual annotations.
- The explanations are AI-generated hypotheses; they do not establish which label is correct.
- Only disagreements are analysed in detail, plus a small seeded control sample; no base rate is estimated.
- The reason vocabulary is our construct, not a validated taxonomy; `Other` absorbs what it does not name.
- The explanation model sees the human label, so it may rationalise toward it; using the same model as the classifier risks self-justification.
- Self-reported confidence ranks examples only, not a probability.
- Where the reference is a majority vote or a single annotator, `Ambiguous reference label` and `Possible annotation error` cannot be confirmed.

## 15. Reproducibility

```
uv run python scripts/05_evaluate.py --config configs/imdb_case_study.yaml
uv run python scripts/06_disagreements.py --config configs/imdb_case_study.yaml
uv run python scripts/07_compare.py --configs configs/imdb_case_study.yaml
uv run python scripts/08_qualitative.py --config configs/imdb_case_study.yaml --settings runs/imdb_case_study/qualitative/settings.yaml --report
uv run python scripts/09_report.py --run-id imdb_case_study
```

Config sha256 `c850923fe261347d38e8fd443a340a6df3cb21e913c4305402d57c608fce4e3e`; model `openrouter/deepseek/deepseek-v4.1-flash-1017`; seed 20261116.

Every number in this report is formatted from a CSV under `results/` (and `qualitative/tables/` when present).


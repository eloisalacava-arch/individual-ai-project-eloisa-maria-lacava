# AI versus human financial-news sentiment classification — study report

| Field | Value |
|---|---|
| Run id | `phrasebank_75agree` |
| Generated | 2026-11-20T11:33:26.913530+00:00 |
| Model | `openrouter/deepseek/deepseek-v4.1-flash-1017` |
| Seed | 20261116 |
| Analysed rows | 300 |
| Categories | negative / neutral / positive |
| Qualitative analysis | included (hypotheses only) |
| Reference | Majority vote of 5-8 finance professionals, restricted to sentences with >=75% annotator agreement. |

#### Executive summary

- 300 observations were classified into 3 categories against the human reference.
- Agreement: accuracy 0.913 (95% CI 0.876–0.940), balanced accuracy 0.913, Cohen's kappa 0.841 (95% CI 0.783–0.897).
- 26 disagreements; the majority class `neutral` is 60.7% of the sample.
- Weakest category: `positive` (recall 0.825 on 80 reference rows).
- Qualitative hypotheses exist for 35 sampled items; the hypothesis most concentrated in disagreements is ambiguous reference label (16.0% above the control).

> Explanations are AI-generated hypotheses, not confirmed causes. The reference is a benchmark, not adjudicated truth, and no causal claim is made.

## 1. Study description

- Task: a financial-news sentiment classifier.
- Decision rule: Judge the sentence from the viewpoint of an investor: would this news positively, negatively, or neutrally affect the company's stock? Consider only the information stated in the sentence.
- Reference: Majority vote of 5-8 finance professionals, restricted to sentences with >=75% annotator agreement.
- Categories and definitions:
  - `negative`: The sentence suggests the news is unfavourable for the company's investors: a fall in value, a loss, or a negative outlook.
  - `neutral`: The sentence carries no clear favourable or unfavourable implication for investors, or its sentiment is not relevant from a financial perspective.
  - `positive`: The sentence suggests the news is favourable for the company's investors: a rise in value, a profit, or a positive outlook.
- Ordered categories: True.
- Sampling: seeded, unstratified (seed 20261116, n=300, unique_text=True).
- Predictions were frozen before the reference was opened (stage 05); the scored copy is `results/frozen_predictions.csv`.

## 2. Agreement with the reference

| Metric | Value | 95% CI | Method |
|---|---|---|---|
| Accuracy | 0.913 | 0.876–0.940 | Wilson |
| Balanced accuracy | 0.913 | 0.877–0.944 | bootstrap |
| Cohen's kappa (unweighted) | 0.841 | 0.783–0.897 | bootstrap |
| Cohen's kappa (quadratic) | 0.846 | 0.769–0.909 | bootstrap |

Accuracy counts exact matches; kappa corrects for chance and depends on the marginal distribution; balanced accuracy averages per-class recall.

Source: `results/t02_accuracy.csv, t03_kappa.csv`

## 3. Confusion matrix

| Human label \ AI answer | negative | neutral | positive | total_human |
|---|---|---|---|---|
| negative | 37 | 1 | 0 | 38 |
| neutral | 4 | 171 | 7 | 182 |
| positive | 3 | 11 | 66 | 80 |
| total_predicted | 44 | 183 | 73 | 300 |

Rows are the human label; columns are the AI answer. The final row and column are marginals.

Source: `results/t04_confusion_matrix.csv`

<!-- figure: confusion_matrix | Confusion matrix, rows = human, columns = AI -->

## 4. Per-category results

| Class | Reference n | Predicted n | Precision | Recall (95% CI) | F1 (95% CI) |
|---|---|---|---|---|---|
| negative | 38 | 44 | 0.841 | 0.974 (0.865–0.995) | 0.902 (0.825–0.962) |
| neutral | 182 | 183 | 0.934 | 0.940 (0.895–0.966) | 0.937 (0.910–0.961) |
| positive | 80 | 73 | 0.904 | 0.825 (0.727–0.893) | 0.863 (0.803–0.916) |

Source: `results/t05_per_class_metrics.csv`

<!-- figure: per_class_recall | Per-category recall with 95% confidence interval -->

## 5. Class distribution

| Class | Human n | Human share | AI n | AI share | Difference (AI minus human) |
|---|---|---|---|---|---|
| negative | 38 | 12.7% | 44 | 14.7% | 2.0% |
| neutral | 182 | 60.7% | 183 | 61.0% | 0.3% |
| positive | 80 | 26.7% | 73 | 24.3% | -2.3% |

Source: `results/t06_class_distribution.csv`

## 6. Disagreement structure

Total disagreements: **26**.

| Bucket | n | Share |
|---|---|---|
| Agree | 274 | 91.3% |
| Adjacent | 23 | 7.7% |
| Opposite | 3 | 1.0% |

The `Share` column is each bucket's share of all 300 analysed rows, not of the 26 disagreements; the buckets sum to 100%.

Adjacency is defined because the categories are ordered; the `Adjacent` share is a severity proxy.

Most frequent disagreement directions:

| Reference | Prediction | n | Share of disagreements |
|---|---|---|---|
| positive | neutral | 11 | 42.3% |
| neutral | positive | 7 | 26.9% |
| neutral | negative | 4 | 15.4% |
| positive | negative | 3 | 11.5% |
| negative | neutral | 1 | 3.8% |

Source: `results/t07_ordinal_errors.csv, t11_disagreement_pairs.csv`

## 7. Observation length (exploratory)

| Group | Median words |
|---|---|
| Agreement | 19.0 |
| Disagreement | 18.5 |

Seeded permutation test p = 0.390; one pre-specified feature, no multiplicity adjustment (exploratory).

Source: `results/t12_text_profile.csv, t12b_length_test.csv`

## 8. Qualitative findings

Explanations below are AI-generated hypotheses, not confirmed causes, and not proof that the classifier or the human made a mistake.

Selection coverage: 25 of 26 disagreements and 10 of 274 agreements were sampled for explanation.

| Group | Selected | Explained | Failed | Explained share |
|---|---|---|---|---|
| Disagreement | 25 | 25 | 0 | 100.0% |
| Agreement | 10 | 10 | 0 | 100.0% |

No multiplicity adjustment is applied to these reason contrasts; they are exploratory hypotheses, and the control sample is small.

A reply can carry up to three reasons, so shares do not sum to 100%.

### Disagreements

| Reason | n | Share of explained | Share of selected | Top directions |
|---|---|---|---|---|
| Insufficient context in the observation | 20 | 80.0% | 80.0% |  (20) |
| Subtle or low-intensity sentiment | 17 | 68.0% | 68.0% |  (17) |
| Needs domain or world knowledge | 16 | 64.0% | 64.0% |  (16) |
| Overlapping category definitions | 11 | 44.0% | 44.0% |  (11) |
| Ambiguous reference label | 4 | 16.0% | 16.0% |  (4) |
| Mixed or conflicting signals | 3 | 12.0% | 12.0% |  (3) |
| Text noise or truncation | 3 | 12.0% | 12.0% |  (3) |

### Control (agreements)

| Reason | n | Share of explained | Share of selected | Top directions |
|---|---|---|---|---|
| Insufficient context in the observation | 9 | 90.0% | 90.0% |  (9) |
| Needs domain or world knowledge | 6 | 60.0% | 60.0% |  (6) |
| Subtle or low-intensity sentiment | 6 | 60.0% | 60.0% |  (6) |
| Overlapping category definitions | 3 | 30.0% | 30.0% |  (3) |
| Text noise or truncation | 2 | 20.0% | 20.0% |  (2) |
| Mixed or conflicting signals | 1 | 10.0% | 10.0% |  (1) |

### Contrast with the control sample

| Reason | Disagreements | Control | Difference (disagreements minus control) |
|---|---|---|---|
| Ambiguous reference label | 16.0% | 0.0% | 16.0% |
| Overlapping category definitions | 44.0% | 30.0% | 14.0% |
| Subtle or low-intensity sentiment | 68.0% | 60.0% | 8.0% |
| Needs domain or world knowledge | 64.0% | 60.0% | 4.0% |
| Mixed or conflicting signals | 12.0% | 10.0% | 2.0% |
| Text noise or truncation | 12.0% | 20.0% | -8.0% |
| Insufficient context in the observation | 80.0% | 90.0% | -10.0% |

A reason common to both groups is not specific to disagreements. The control sample is small and its own explanations are hypotheses.

### Illustrative examples

Illustrative examples are omitted from this public copy to avoid redistributing source text. The original examples remain in the author's working tree and are not part of this repository.

Source for the aggregate qualitative tables: `results/qualitative/tables/q02_patterns.csv`, `q05_coverage.csv`, `q07_reason_contrast.csv`.

## 9. Interpretation

- The model agreed with the human reference on 0.913 of 300 analysed rows (kappa 0.841); balanced accuracy 0.913 shows this is not purely a dominant-class effect.
- The majority reference class `neutral` is 60.7% of the sample, which raises chance agreement and makes kappa harder to compare across studies.
- Recall ranged from 0.825 for `positive` to 0.974 for `negative`; the lower figure means that category's reference rows were matched less often. It does not say how many rows that is, or why.
- The AI predicted `positive` 2.3 percentage points less often than the human reference did.
- Of all 300 analysed rows, 7.7% (23) were adjacent disagreements and 1.0% (3) were opposite; the adjacent share gauges how severe the errors are.
- Qualitatively, ambiguous reference label was the hypothesis most concentrated in disagreements (16.0% above the control).
- Text noise or truncation appeared in both disagreements and the control, so it is not specific to disagreement.

## 10. Conclusions

- On this sample the model agreed with the reference on 0.913 of rows, with kappa 0.841.
- The category most often missed was `positive` (recall 0.825).
- Qualitatively, ambiguous reference label was the hypothesis most concentrated in disagreements; it, and every other hypothesis here, remains unconfirmed.
- These are descriptive associations within one sample; no causal claim is made, and a disagreement is not a confirmed model error.

## 11. Limitations

- The reference is a benchmark label, not an adjudicated gold standard; a disagreement is not a confirmed model error.
- Accuracy and recall use Wilson intervals; kappa, balanced accuracy, precision and F1 use a seeded percentile bootstrap.
- All intervals are conditional on this sample; the sample is not stratified.
- No causal claim is made. Statements are descriptive associations within one sample.
- Temperature is not controlled and model versions drift.
- The sample is drawn from the >=75% annotator-agreement subset; sentences with weaker agreement are excluded by construction, so agreement with the majority label is plausibly overstated relative to the full corpus.
- Only 300 of the 3,453 available 75%-agreement sentences were sampled (unstratified, seed 20261116).
- Annotators came from one institution (Aalto University School of Business); agreement should be read with that in mind.
- The published agreement rate is item-level, not a pairwise kappa, so it is not directly comparable to Cohen's kappa.
- The benchmark is a majority verdict.
- Domain is English financial news from OMX Helsinki listed companies, collected before 2014.
- Licence is CC BY-NC-SA 3.0: non-commercial use only; data is not redistributed with this agent.
- The explanations are AI-generated hypotheses; they do not establish which label is correct.
- Only disagreements are analysed in detail, plus a small seeded control sample; no base rate is estimated.
- The reason vocabulary is our construct, not a validated taxonomy; `Other` absorbs what it does not name.
- The explanation model sees the human label, so it may rationalise toward it; using the same model as the classifier risks self-justification.
- Self-reported confidence ranks examples only, not a probability.
- Where the reference is a majority vote or a single annotator, `Ambiguous reference label` and `Possible annotation error` cannot be confirmed.

## 12. Reproducibility

```
uv run python scripts/05_evaluate.py --config configs/case2_phrasebank.yaml
uv run python scripts/06_disagreements.py --config configs/case2_phrasebank.yaml
uv run python scripts/07_compare.py --configs configs/case2_phrasebank.yaml
uv run python scripts/08_qualitative.py --config configs/case2_phrasebank.yaml --settings runs/phrasebank_75agree/qualitative/settings.yaml --report
uv run python scripts/09_report.py --run-id phrasebank_75agree
```

Config sha256 `7bf1eaffbfb5b743d6d8a2477f991624c8c49104d8048988b05cd8b88e6e3eb9`; model `openrouter/deepseek/deepseek-v4.1-flash-1017`; seed 20261116.

Every number in this report is formatted from a CSV under `results/` (and `qualitative/tables/` when present).


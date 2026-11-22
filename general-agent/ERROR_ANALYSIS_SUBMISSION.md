# Error Analysis — Individual AI Project

**Project:** `general-agent`, a reusable AI classification and evaluation agent.

**Scope:** the individual project. Earlier group-project evidence is context only and is prefixed `GP-`; individual errors are prefixed `GA-`. The 23-entry register covers phases 0–6. Later `SPEC.md` corrections are reported separately and are not added to that count.

## 1. Executive summary

The agent classifies text with an AI model and compares predictions against a sealed human benchmark: the model sees only an identifier and text, predictions are frozen and hashed before unsealing, and a five-row pilot is bound to the model and prompt. It recorded 23 implementation mistakes in phases 0–6. Twelve were caught by a test or reconciliation check, eight by a real run or visible output, two by code review, and one by a catalogue check. The 23 registered corrections preceded the final tables. One defect remains: the `t08` row-level severity label (§8). Three interface corrections sit outside that count. The case studies produced 6 and 26 disagreements; no confirmed model error is established. The individual agent has not undergone independent external verification.

## 2. Terms and evidence standard

An **implementation error** is a defect in code, configuration, accounting or documentation that produced, or could have produced, a wrong artefact. A **workflow error** is a rule one stage assumes but another does not materialise. A **classification disagreement** is a prediction that differs from a benchmark label; it is not a confirmed model error, because confirmation requires adjudication or per-annotator labels that do not exist here. `t10_error_status.csv` sets `confirmable_with_this_data = False` for both studies.

Evidence is **[Verified]** when a named file or test supports it, **[Internal]** when the check is same-system rather than independent, **[Interpretation]** when it is my reading, and **[Not estimable]** when the data cannot support one. The project answers two questions: whether the pipeline was correct, which is largely checkable, and whether the model was right, which is not.

## 3. Where the errors occurred

Errors clustered at stage boundaries. The audit wrote benchmark example values into `decisions.json` (`GA-M8`). The pilot gate recorded the model and prompt but not the pilot file (`GA-M10`). Budget and cost totals came from the wrong logs (`GA-M13`, `GA-M16`, `GA-M17`). Evaluation read the full source rather than the sealed sample (`GA-M18`), and two stages repeated the same merge logic, so a `label` collision recurred (`GA-M19`, `GA-M23`). Configure and audit rebuilt the manifest and erased later records (`GA-M9`). Two metric boundary values were mishandled (`GA-M20`, `GA-M21`).

Building the interface later exposed three further defects: a blind-stage policy the audit had validated but the stage did not materialise, a hard-coded zero in the evaluation log, and a single-study report carrying a cross-dataset heading (`SPEC.md` v0.5, outside the register). These were not random slips but failures to carry a decision from where it was specified to where it was used.

## 4. The 23-entry register: consequential errors

Eleven entries carry the substantive risk; the remaining twelve are minor. The table pairs each group with its detection route and correction.

| ID | Stage | What went wrong | Detected by | Correction and evidence |
|---|---|---|---|---|
| `GA-M8` | 01 audit | Benchmark example values written to `decisions.json`; a poison marker escaped the seal | Poison-marker test | Examples redacted; test passes |
| `GA-M10` | 03 pilot | Approval recorded model and prompt hash but not the pilot-file hash | Code review | SHA-256 binding; test passes |
| `GA-M9` | 01–02 | Re-running configure or audit rebuilt the manifest and erased later stage records | Regression test | Stage records merged; test passes |
| `GA-M13`, `GA-M16`, `GA-M17` | 03–04 | Budget counted per log file; cost reset on a no-op resume; pilot cost omitted | Pre-pilot arithmetic, manifest refresh, cost reconciliation | Logs are the source; manifest audit finds no violation |
| `GA-M18` | 05 evaluate | Read the full 3,453-row source instead of the sealed 300 | Real evaluation run and row counts | Sealed-only read; `t01` 300 → 300; 13/13 checks |
| `GA-M19`, `GA-M23` | 05–06 | Prediction and benchmark both had `label`; stage 06 repeated the merge and crashed | Real PhraseBank runs | Rename, assert columns; zero rows dropped |
| `GA-M20`, `GA-M21` | 05 evaluate | Weighted kappa applied agreement weights as disagreement; F1 returned `NaN` at zero precision | Test design and unit test | Correct formula; F1 = 0; tests pass |

**Why these mattered.** `GA-M18` and `GA-M20` could have changed a reported conclusion: the first would have scored 300 predictions against 3,453 reference rows; the second would have reported kappa = 1 for maximally disagreeing data. `GA-M8` and `GA-M10` could have broken the evaluation barrier. `GA-M9` and the budget group could have destroyed provenance or bypassed a cap.

Three causes recur: a policy validated in memory but never materialised; duplicated logic that left one call site unprotected; and summary text hard-coded rather than generated from the table it described. These design failures explain why the same classes of defect returned across phases.

**Earlier group-project context.** The design responds to `GP-M7`: an earlier McNemar table used 10/5 instead of 1/5 (p = 0.302 instead of 0.219), and the error survived until an independent pass recomputed it. That is why this agent freezes predictions before opening the reference. Group-project numbers are context only.

**Later corrections outside the register.** Building the interface produced three method-neutral corrections in `SPEC.md` v0.5: stage 02 now materialises the ID-intersection and label-exclusion policies the audit validated (`test_blind.py`); the evaluation log reports the real failed-prediction count instead of a hard-coded zero (`test_evaluation_log_reports_the_real_failure_count`); and a single-study report no longer carries a cross-dataset heading (`test_report_reuse.py`). v0.10 and v0.11 are documentation-only.

**The remaining twelve.** `GA-M1`–`GA-M7`, `GA-M11`–`GA-M12`, `GA-M14`–`GA-M15` and `GA-M22` are minor input, summary, parsing and genericity issues; none changed a reported statistic.

## 5. Detection-method breakdown

| Detection route | Count of 23 | What it caught |
|---|---|---|
| Unit, regression, leakage or genericity test | 9 | Encoded expectations, including boundary values |
| Reconciliation or arithmetic check | 3 | Row counts, cost totals and budget arithmetic |
| Real run or visible output | 8 | Integration and real-data defects |
| Code review | 2 | Gate logic and guard behaviour |
| Model-catalogue check | 1 | A stale configured price |

Tests were the largest route, but eight defects required a real run and three reconciliation. Tests plus reconciliation account for 12 of 23 (52%), a narrow majority. A test catches only what it encodes; integration caught the boundary defects, where the register clusters.

## 6. How the corrections were verified

Each selected correction has a named guard: poison-marker (`GA-M8`), pilot-file (`GA-M10`), manifest-regression (`GA-M9`), kappa and F1 (`GA-M20`/`GA-M21`) tests. `GA-M18`, `GA-M19` and `GA-M23` are held by `t01` and post-merge column assertions: 200 → 200 and 300 → 300, with zero dropped or imputed rows. Logs report 15/15 and 13/13 checks, including `benchmark_opened_after_freeze` and `disagreement_count_reconciles` (6 = 200 − 194; 26 = 300 − 274). `test_manifest_accounting.py` finds no cap or spend violation.

These are internal controls. Reproduction tests recompute the IMDb kappa targets through a separate code path in the same project, a regression check, not an independent audit. The individual agent has not undergone independent external verification. The only independent evidence belongs to the earlier group project and does not cover PhraseBank, the qualitative stage or the comparison.

## 7. Classification disagreements

A disagreement is a difference from a benchmark, not proof that the model erred. Both studies report zero confirmed model errors; every disagreement is marked unconfirmed by construction.

| Study | n | Reference | Agree | Disagreements | Adjacent | Opposite | Confirmed errors |
|---|---:|---|---:|---:|---:|---:|---:|
| IMDb | 200 | One manual annotator | 194 | 6 | 0 | 6 | 0 |
| PhraseBank | 300 | Majority of 5–8 annotators, ≥75% agreement | 274 | 26 | 23 | 3 | 0 |

**IMDb.** The three-way decomposition is more informative: 184 rows agree across all three quantities; 1 is a consensus disagreement where manual annotation and proxy agree and the AI differs; 10 are disputes where the AI sides with the manual annotation; 5 with the proxy; 0 with neither. Only 1 of the 6 is a clean "both references agree, AI differs" case. Manual annotation and proxy agree on 0.925 (kappa 0.850); AI and proxy on 0.945 (kappa 0.890).

**Financial PhraseBank.** Of the 26 disagreements, 23 are one-step and 3 two-step. The pairs are positive→neutral 11, neutral→positive 7, neutral→negative 4, positive→negative 3, negative→neutral 1. Positive-class recall is 0.825 and negative-class precision 0.841, so disagreement concentrates in the positive and negative classes, not the neutral majority.

**Where they concentrate.** Against the manual annotation, the pre-specified `ambiguous_or_complex` composite has 6 disagreements in 50 rows versus 0 in 106: risk difference +12.0 pp (95% CI 4.7–23.8), Fisher p = 0.000875. Against the star-derived proxy the same composite is +4.1 pp (−2.0 to +14.4), p = 0.328, and the verdict is inconclusive. The reference changes the apparent result. Difficulty flags overlap, so subgroup n values do not sum.

**Qualitative hypotheses.** The IMDb qualitative pass attributed "ambiguous reference label" to 5 of 6 disagreements (83.3%) and 0 of 10 agreements. These are hypotheses: the explanation model saw the human label, is the same model that classified, the control is small, and no multiplicity adjustment applies. They suggest where to look, not what caused the disagreement.

## 8. Unresolved defect: `t08` severity labels

A new inconsistency exists in `runs/imdb_case_study/results/t08_disagreements.csv`: for the binary IMDb task the `severity` column says `adjacent` in all six rows. Summary tables `t07_ordinal_errors.csv` and `comparison/tables/cross_dataset_severity.csv` classify all six as `opposite`. In binary data, distance 1 is both the only step and the maximum distance; `metrics.py` and the `t07` builder handle that correctly, but `analysis.py` line 299 assigns `adjacent` whenever distance equals 1. No metric or count changes; the defect is confined to the `t08` row label. The original CSV is unedited. Until a fix regenerates it, `t07` and the cross-dataset table are authoritative for severity.

## 9. Remaining limitations

| Risk | Evidence | Consequence |
|---|---|---|
| No independent verification | `VERIFICATION.md` §1, §5 | Internal consistency only |
| No reliability estimate | `codebook.md` §7; `MISTAKES.md` Part C | Annotation noise unmeasured |
| Reference is not truth | `data/README.md` §5; `SPEC.md` §6 | Disagreement is not model error |
| Aggregate class counts seen | `evaluation_log.txt` | Weak aggregate leakage not excluded |
| Model conditions uncontrolled | `SPEC.md` §2; `README.md` | Run-to-run stability weakly evidenced |
| Checks documented but not implemented | `SPEC.md` §4 | McNemar, Holm and minimum detectable difference absent |
| Qualitative self-justification | `qualitative/narrative_report.md` | Hypotheses, not causes |
| Legacy schema and hash drift | `VERIFICATION.md` §3–5 | Audit friction; no budget violation |

The largest residual risk is the absence of independent verification: every guard was written by the process it checks. Benchmark uncertainty is not a technicality: the IMDb secondary label is star-derived, the PhraseBank reference is a majority verdict on the ≥75% subset, and neither is adjudicated truth. The evaluation can show where the model differs, not who is right. I am not sure how much residual disagreement is annotation noise.

## 10. Implications for trusting AI agents

The consequential errors were process failures, not arithmetic failures. Plausible output came from a pipeline that had not carried its rules through, and fluency made the gap hard to see. Three lessons follow. Tests are necessary but bounded: they catch what they encode, while integration and reconciliation catch what crosses stages. Gates must bind identity, not just intent: an approval that does not name the pilot file is not a gate. Reports and logs must be generated from the tables they describe, because hard-coded prose is where wrong numbers survive.

For relying on agents, use them for bounded work with explicit acceptance checks, keep human authority at evaluation design and release, and never let an agent be the only verifier. The missing independent pass separates internally consistent from externally checked output. I would not present these disagreements as model errors, nor treat the agent's self-checks as an audit.

## Appendix: Evidence map

| Claim | Evidence |
|---|---|
| 23 implementation errors and detection routes | `MISTAKES.md` Part A |
| Blind/seal design and poison-marker leak | `SPEC.md` §3; `MISTAKES.md` `GA-M8`; `tests/test_leakage.py` |
| Pilot hash binding | `MISTAKES.md` `GA-M10`; `tests/test_classify.py` |
| Full-source evaluation defect | `MISTAKES.md` `GA-M18`; `t01_join_and_sample.csv`; `evaluation_log.txt` |
| `SPEC.md` v0.5 corrections | changelog v0.5; `test_blind.py`; `test_evaluate.py`; `test_report_reuse.py` |
| Manifest overwrite | `MISTAKES.md` `GA-M9`; `tests/test_blind.py` |
| Budget and cost accounting | `MISTAKES.md` `GA-M13`, `GA-M16`, `GA-M17`; `tests/test_classify.py`; `tests/test_manifest_accounting.py` |
| Join collision and recurrence | `MISTAKES.md` `GA-M19`, `GA-M23`; `t01_join_and_sample.csv`; `evaluation_log.txt` |
| Statistical boundary defects | `MISTAKES.md` `GA-M20`, `GA-M21`; `tests/test_metrics.py` |
| Disagreements, three-way and subgroup results | `t07`, `t10`, `t11`, `t14`, `t15`, `t16`, `t12b` |
| Cross-dataset severity and references | `comparison/tables/cross_dataset_severity.csv`, `cross_dataset_references.csv` |
| Qualitative hypotheses | `qualitative/narrative_report.md` |
| Verification limits and unimplemented checks | `VERIFICATION.md` §1–5; `SPEC.md` §4 |
| Group-project context | `mistakes.md` `GP-M7`, Part F; `results/verification_findings.md` |

*No causal claim is made. All comparisons are associations on the analysed samples.*
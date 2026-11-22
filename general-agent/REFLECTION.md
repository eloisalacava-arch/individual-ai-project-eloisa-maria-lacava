# Reflection

**Project:** general-agent — a reusable AI classification and evaluation agent
**Scope:** the individual project (the agent, its two case studies, and the
cross-dataset comparison). The earlier group project (a `workflow_*` pipeline, an
`agent_*` re-analysis, and a separate verification pass) is cited only as
historical evidence.

Throughout this document **verified** means the claim can be checked in a named
file; **interpretation** is my reading of those files; **uncertainty** marks
where I am not sure. I have not relied on recollection, and I have not described
the IMDb star-derived dataset labels as a second independent human annotator.

---

## 1. Design decisions and iterative improvement

The decision that shapes everything else is to enforce the separation of
prediction from evaluation structurally rather than by intention. The classifier
reads only a blind file containing `id` and `text`; the reference labels live in
a separate sealed file rendered unreadable (`chmod 000`) during classification;
and predictions are frozen and hashed before the sealed reference is opened.
[verified: `SPEC.md` §3; `README.md`] Six leakage layers are documented,
including an import-boundary test and a poison-marker test that plants a unique
token in benchmark metadata and checks it never reaches a prompt or log.
[verified: `tests/test_leakage.py`]

I built the agent in phases and tagged each one (v0.1–v0.7). [verified:
`SPEC.md` changelog] Three of those changes were not features but method-neutral
corrections found only once a later surface was built: the blind stage did not
materialise the ID-intersection and label-exclusion policies the audit had
validated, so the blind input and the sealed benchmark could disagree; the
evaluation log reported a hard-coded zero failed predictions; and the report
heading ignored the number of studies. [verified: `SPEC.md` v0.5 changelog] That
pattern — a later surface exposing an earlier inconsistency — is why I kept a
changelog and a mistake register.

The design rule I converged on is to make correctness structural.
Freeze-before-read, sealed-only evaluation, row-count reconciliation, and tests
that fail closed caught defects a reader would likely miss. Most of the 23
implementation mistakes recorded for this agent were detected by a test or a
reconciliation check before any number was reported. [verified: `MISTAKES.md`
Part A]

**Interpretation.** The agent's value is not that it is "correct", but that the
evaluation protocol is explicit, bounded, and checkable, and that it refuses to
score against a reference it has not frozen first.

## 2. Mistakes, failed assumptions, debugging and corrections

I find it more useful to group the mistakes than to list them. [verified:
`MISTAKES.md`; `../mistakes.md`]

- **Statistical**: a McNemar table mis-tabulated (10/5 instead of 1/5); a
  Cohen's kappa recorded as 0.905 instead of 0.850; an odds ratio reported under
  complete separation that should have been `not_estimable`. [verified:
  `../mistakes.md` M1, M2, M7]
- **Boundary cases**: a weighted kappa with an inverted sign, and an F1 that
  returned `NaN` when precision was zero; both were caught while writing the
  tests, before use. [verified: `MISTAKES.md` M20, M21]
- **Provenance**: a stale free-tier prediction file with a generic name, and a
  stale "primary model" line sitting above a changelog that recorded the model
  actually used. [verified: `../mistakes.md` D1, D2]
- **Human-only gaps**: no intra-annotator reliability check, although the
  annotation codebook required one. [verified: `../mistakes.md` U1]

The falsified assumptions are more instructive than the arithmetic slips:

1. *The dataset label is truth.* For IMDb the label is star-derived from ratings,
   not a reading of the review text. [verified: `../data/README.md` §5]
2. *One annotator is a reliable reference.* The manual annotation has no
   reported reliability, and the disagreement analysis shows most apparent model
   errors are not clear-cut. [verified: `../mistakes.md` U1, U2]
3. *Temperature is controllable.* The CLI used has no temperature flag; the
   value is recorded as `not_requested` and the report may not claim otherwise.
   [verified: `SPEC.md` §2]
4. *The free tier suffices.* Quota exhaustion forced a switch to a single paid
   endpoint, which is what created the stale-file and stale-summary problems.
   [verified: `../mistakes.md` U6, D1, D2]

**A methodological correction to my own earlier framing.** The IMDb case study
involves three distinct quantities, and they are not three annotators: the AI
prediction; a manual annotation (one person's reading of the text); and a
star-derived proxy label (derived from ratings, not from reading the text). An
earlier generated narrative called the latter two "the two human references",
which is imprecise: one is a human annotation and the other is a non-human proxy.
I corrected the wording in the report generator and regenerated the IMDb and
comparison reports, so both now name the pair as "manual human annotation"
versus "star-derived IMDb proxy label" and state explicitly that a secondary
reference may be a proxy rather than a second human annotation. [verified:
`reports/imdb_case_study/study_report.md`; `reports/comparison/report.md`;
`src/general_agent/evaluate/subgroups.py`;
`runs/imdb_case_study/results/t15_three_way_agreement.csv`]

**Debugging.** The correction of the McNemar table is the worked example: it was
detected by recomputing the 2×2 tables from the joined data, then repaired at the
script, the table, the log, and the report, with a written record proving that no
other result changed. [verified: `../mistakes.md` Part F] The lesson I take is
that a correction needs a scope statement and a "what did not change" proof, not
just a fix.

**Uncertainty.** I am not sure how much of the residual IMDb error is annotation
noise versus model error, because no reliability estimate exists.

## 3. AI autonomy versus human oversight

The boundary between agent and human was engineered rather than hoped for. A
genuine ambiguity becomes an exit code and a named config field rather than an
interactive prompt, so re-runs are non-interactive. [verified: `SPEC.md`
decision 7] The five-row pilot must be approved before the full run, and the
approval binds the model name and the prompt-template hash, so changing a class
definition invalidates it. [verified: `SPEC.md` decision 12] Who approves it
depends on the flow. The default three-step (Simple) flow approves the pilot
**automatically** once every row parses to a declared category — technically
valid — and pauses to ask only when a pilot row fails; the seven-step (Advanced)
flow keeps the same gate **manual**, so the reviewer reads the five rows and
clicks *Approve pilot* with an optional note. [verified: `README.md` (interface
paragraph); `app/README.md` §Two flows, one engine; `app/pipeline.py`;
`acceptance/GUIDE.md` step 6] In both flows one authorization covers the bounded
run — the automatic approval is a pre-committed decision inside that
authorization, not blanket consent — changing the model or raising a cap needs a
new one, and the qualitative stage has its own three-item pilot, also
auto-approved only when clean. [verified: `app/README.md`; `app/pipeline.py`]

**Simplifying the interface from seven steps to three.** The first interface
exposed the engine faithfully: seven screens, one per stage, each asking the user
to confirm a decision the engine could already make. User testing of that flow
showed it demanded human intervention well beyond the points of irreversible
cost or epistemic risk — the audit, the config defaults and the provider
catalogue could already propose the columns, categories, model and caps — and
every extra screen was a place to stall. I made a three-step flow
(Upload → Review → Analyze & Download) the default and moved the seven-step flow
behind Advanced mode, without changing the engine or its safeguards: the same
seal, pilot gate, freeze-before-read and caps. The trade-off is deliberate.
Autonomy and usability rise, because one authorization runs the bounded analysis
and the common path is three screens; what falls is *visibility*, because a user
who stays in Simple mode sees fewer intermediate decisions and corrects a bad
proposal in Review rather than stage by stage. I kept the safeguards by making
the automatic pilot approval narrow (only a cleanly parsed pilot) and by leaving
the manual gate one click away, but I am not sure a reviewer who never opens
Advanced mode reads the five pilot rows as closely as the manual flow once forced
them to. [interpretation; the two flows are documented in `README.md`,
`app/README.md`, and `app/state.py`]

Within that frame the agent decided a great deal by itself. In the earlier group
project a decision log records 26 choices made without asking, from the primary
reference to the multiple-testing family. [verified:
`../results/agent_autonomous_decisions.md`] The individual agent likewise
proposes columns, category names, mappings, model, and caps, and raises only the
items that could corrupt the evaluation. [verified: `app/README.md`]

**Interpretation.** The right balance is "the agent proposes and executes; a
human authorises at the points of irreversible cost or epistemic risk". The
evidence is negative but strong: in the group project the agent documented a
conceptual confusion between a manual error and a label-versus-manual dispute
and still left a wrong McNemar cell in place until an independent pass
recomputed it, and it typed a confidence interval into a report from console
output until the value was moved into a table. [verified: `../mistakes.md` M5,
M6, M7, Part E] Autonomy without a review gate reproduced plausible-looking
errors.

The individual project institutionalised several gates — sealed benchmark, pilot
approval, freeze-before-read, and an acknowledgement before text is sent to a
provider. I should be precise about what those are: the individual agent has
**no independent external verification**. The separate verification pass belongs
to the earlier group project, where it recomputed the numbers from scratch and
also found and fixed two bugs in its own first draft. [verified:
`../results/verification_findings.md`] The individual project's safeguards are
internal controls, not an audit.

## 4. Reliability, benchmark uncertainty, self-justification, generalizability

**Reliability.** The agent is reliable in the narrow, checkable sense: seeded
sampling, hashed artefacts, per-call attempt logs, and a per-stage manifest.
[verified: `README.md`; `runs/*/manifest.json`] Temperature and the serving
endpoint are not controlled, so run-to-run stability is only weakly evidenced;
the five-row pilot reproduced. [verified: `SPEC.md` §2; `MISTAKES.md` Part C]

**Benchmark uncertainty.** For IMDb the primary reference is a single manual
annotation and the secondary reference is the star-derived proxy; neither is
adjudicated truth. [verified: `configs/imdb_case_study.yaml`] For PhraseBank the
reference is a majority verdict of 5–8 annotators restricted to the ≥75%
agreement subset, so agreement is plausibly overstated relative to the full
corpus, and per-annotator labels are not released. [verified: `SPEC.md` §6;
`configs/case2_phrasebank.yaml`] The project's response is to label every
disagreement **unconfirmed** and to report zero confirmed model errors.
[verified: `runs/*/results/t10_error_status.csv`] The three-way decomposition
makes the consequence concrete: of the six IMDb disagreements with the manual
annotation, only one is a row where the manual annotation and the proxy label
agree; the other five are rows where those two references differ, so a
disagreement with the manual annotation is not a clean count of model mistakes.
[verified: `runs/imdb_case_study/results/t15_three_way_agreement.csv`]

**Qualitative self-justification.** The explanation model sees the human label
and is the same model that produced the classifications; the control sample is
ten agreements and no multiplicity adjustment is applied. [verified:
`app/README.md`; report limitations] This is a real risk: a model asked to
explain a disagreement while looking at the human label can rationalise toward
it. The mitigations are framing (hypotheses, not causes; a versioned vocabulary;
deterministic counting) and a control group. **Interpretation:** I read the
qualitative tables as hypotheses about what to inspect next, not as evidence of
cause.

**Generalizability.** Two convenience samples, one model string, one prompt, and
two domains; pre-training contamination cannot be excluded; kappa is sensitive to
class prevalence; and the group project's power analysis puts the minimum
detectable difference at roughly 19–29 percentage points against the
straightforward reviews. [verified: `MISTAKES.md` Part C;
`../results/verification_findings.md` §5, §8; `comparison/report.md`] The
hypothesis verdict follows directly. Against the pre-specified star-derived
reference, the plan's `ambiguous_or_complex` composite is **inconclusive** (risk
difference +4.1 pp, 95% CI −2.0 to +14.4, p = 0.328); against the manual
annotation it is **supported** (+12.0 pp, p = 0.001). `mixed` is the only
pre-specified flag whose interval excludes zero against the star-derived label,
and it is marginal after correction. [verified:
`runs/imdb_case_study/results/t14_subgroup_tests.csv`;
`reports/imdb_case_study/study_report.md`]

## 5. Lessons learned and what I would do differently

Lessons from using AI-assisted development, each with an instance:

1. Plausible output is not verified output — an invalid odds ratio was formatted
   like a real result. [verified: M1]
2. An agent can document a problem and still not fix it — the label/dispute
   confusion left a wrong cell in place. [verified: M6, M7]
3. Silent state reuse produces plausible output with the wrong denominator.
   [verified: M9]
4. Boundary conditions need an explicit rule (`not_estimable`), not a coding
   exception. [verified: M1]
5. Hard-coded numbers are a governance failure; generating the report from the
   tables removed the class of error. [verified: M5]
6. Self-review is not independent review — the separate pass found bugs in its
   own first draft. [verified: M8, M9]

What I would do differently:

1. Collect inter- and intra-annotator reliability *before* analysis, so the
   benchmark's uncertainty is measured rather than asserted. [verified: U1]
2. Fix the primary reference up front and justify it: the agent's IMDb case study
   uses the manual annotation as primary and the star-derived label as
   secondary — defensible, but a retrofitted choice; the reports now name the
   pair as annotation-versus-proxy, but choosing it after the fact remains a
   weakness. [interpretation]
3. Quarantine stale artefacts by naming, and keep a single source of truth for
   the model and prompt. [verified: D1, D2]
4. Keep the method record and README current; several status lines lagged the
   code, and I corrected them in this task. [verified: this reflection]
5. Prefer one verified analysis to three parallel ones unless the verification is
   structurally independent. The group project ran a workflow, an agent
   re-analysis, and a separate verification pass.
6. Never let an agent be the only verifier. The gates in the individual project
   are a good start, not an audit.

## 6. Evidence map

| Claim | Evidence file |
|---|---|
| Blind/seal/freeze design and leakage layers | `SPEC.md` §3; `README.md` |
| 23 implementation mistakes and how they were detected | `MISTAKES.md` |
| Statistical mistakes and the McNemar correction record | `../mistakes.md` Parts A, F |
| Falsified assumptions (labels, reliability, temperature, quota) | `../mistakes.md` U1–U6, D1–D3; `../src/workflow_00_model_and_prompt.md` |
| Autonomous decisions versus human gates | `../results/agent_autonomous_decisions.md`; `app/README.md`; `SPEC.md` decisions 7, 12 |
| Three-step default, seven-step Advanced mode, and per-flow pilot approval | `README.md`; `app/README.md`; `app/state.py`; `app/pipeline.py`; `acceptance/GUIDE.md` |
| Separate verification pass (group project only) | `../results/verification_findings.md` |
| Benchmark uncertainty and the three-way decomposition | `runs/imdb_case_study/results/t10_error_status.csv`, `t15_three_way_agreement.csv`; `comparison/report.md` |
| Subgroup verdict | `runs/imdb_case_study/results/t14_subgroup_tests.csv`; `reports/imdb_case_study/study_report.md` |
| Qualitative self-justification and control sample | `app/README.md`; `runs/imdb_case_study/qualitative/tables/q05_coverage.csv` |

*Every statement is a descriptive association on the analysed samples; no causal
claim is made.*

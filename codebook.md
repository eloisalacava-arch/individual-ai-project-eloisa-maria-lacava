# codebook.md — Manual Annotation Codebook

**Version 1.0 — IMDb Sentiment Error Analysis**

Purpose: define the manual annotation of 200 IMDb reviews into difficulty
categories, *before* any model evaluation, as required by plan.md §4. These
categories exist to test the hypothesis in plan.md §3: *"AI classification
accuracy will be lower for ambiguous or linguistically complex reviews than for
straightforward reviews."*

Status labels: **[Decision]** = an analysis choice recorded at approval;
**[External]** = from published guidance (e.g., Maas et al. 2011 for the label
scheme), not verifiable from local files.

---

## 0. General annotation rules

- **Annotate one review at a time, in the fixed random order supplied by the
  sampling script. Do not re-read or re-annotate earlier reviews except for the
  reliability check in §7.**
- **Blind procedure (§6):** you must never see the dataset `label` column, any
  star rating, or any AI prediction while annotating. Only the review text is
  shown.
- **Annotate the reviewer's overall stance on the movie**, not the reviewer's
  stance on other people's opinions, the film industry, or other reviews.
  Quoted praise of the movie inside a review that panned it is *not* a positive
  stance.
- **Text presentation:** `<br />` tags are replaced with line breaks before
  annotation (the same cleaning later applied to model input). **[Decision]**
  Everything else (spelling, entities) stays as-is; do not correct grammar when
  judging it.
- **When in doubt:** record your best judgment *and* raise the uncertainty flag
  (§5). Never leave a field blank. If you are not sure, say so in the
  uncertainty field — per AGENTS.md.

## 1. Operational definitions

### 1.1 Sarcasm / irony (`sarcasm` = 1)
The reviewer's **surface wording systematically inverts their intended
meaning**: positive-sounding words are used to convey a negative verdict, or
(less commonly) vice versa.

- **Include if:** the review praises the movie in terms the reviewer clearly
  does not mean ("a masterpiece of tedium"), uses sustained irony, or signals
  inversion through exaggeration + a contradicting conclusion.
- **Exclude if:** the wording is negative and the meaning is negative, even if
  extremely hyperbolic ("the worst film ever made" is literal negativity, not
  sarcasm). Sarcasm about *another reviewer* or about IMDb reviewers does not
  count unless it extends to the movie. A single ironic phrase inside an
  otherwise plain review still counts (the category is binary, not
  proportional).

### 1.2 Mixed opinions (`mixed` = 1)
The review contains **both at least one distinct, clearly positive evaluation
and at least one distinct, clearly negative evaluation** of different aspects,
with the reviewer genuinely on the fence *or* explicitly trading them off.

- **Include if:** e.g., "the cinematography is stunning, but the script is a
  mess" with both claims defended, even if the reviewer ends with a slight
  lean.
- **Exclude if:** one side is a trivial aside ("only flaw: the runtime") that
  the reviewer explicitly dismisses. **Threshold [Decision]:** the minor aspect
  must receive its own sentence or defended claim, not a passing adjective. A
  review that is 90% praise + one dismissed quibble is *not* mixed.

### 1.3 Ambiguity (`ambiguous` = 1)
The reviewer's **overall stance cannot be determined** from the text: the
review fails to commit, contradicts itself on its verdict, or explicitly states
indecision ("I honestly can't decide if I liked it").

- **Include if:** hedged verdict ("I guess it was… fine?"), mutually cancelling
  conclusions, or stated confusion about their own reaction.
- **Exclude if:** the stance is merely *mixed but decided* (that is `mixed`),
  or the stance is clear but the *reasons* are unclear. Ambiguity is about the
  **verdict**, not the argument quality.
- **If you truly cannot decide the stance at all:** still choose the side it
  leans toward using the tie-break in §4.3, set `uncertainty_sentiment` = 2,
  and set `ambiguous` = 1.

### 1.4 Complex language (`complex` = 1)
The review's **linguistic form makes comprehension demanding**: long
multi-clause sentences, uncommon/abstract vocabulary, dense or academic
phrasing, sustained metaphor, or extended abstract/philosophical discussion of
the film's meaning.

- **Include if:** most sentences are long and multi-clausal; vocabulary is
  markedly above everyday register; the review reads like an essay rather than
  a comment.
- **Exclude if:** the review is long but simply worded (length alone is not
  complexity); it is technical *film jargon* only (terms like "mise-en-scène"
  alone do not qualify); or it is complex only because of typos/artifacts.
- **Objective proxy [Decision]:** after annotation, a script will compute
  median sentence length and word length for each of the 200 reviews and
  record it, so the human judgment can be cross-checked against a measurable
  proxy. This number comes from a script → `results/`.

### 1.5 Straightforward (`straightforward` = 1, **derived, not annotated**)
A review is straightforward when **all four flags above are 0**: it expresses a
single clear direction of opinion in plain language. This is the reference
class for the hypothesis. It is computed from the four flags, never annotated
directly, so it cannot conflict with them.

## 2. Inclusion / exclusion summary table

| Category | Include when | Exclude when |
|---|---|---|
| `sarcasm` | Surface wording inverts intended stance | Hyperbole that is literally negative/positive |
| `mixed` | ≥1 defended positive aspect AND ≥1 defended negative aspect | One side is a dismissed passing remark |
| `ambiguous` | Verdict cannot be determined / not committed | Mixed but decided stance |
| `complex` | Long multi-clause sentences, uncommon vocab, essay-like density | Long but plain wording; film jargon alone |
| `straightforward` | (derived) all four flags = 0 | — |

## 3. Examples

All examples below are **synthetic and illustrative — none are taken from the
dataset** (using real reviews here would risk label peeking). During
annotation, judge each real review by the definitions, not by resemblance to
one example.

- **Sarcasm:** *"What a triumph — two hours of my life I'll never get back. The
  'thrilling' chase scene was so thrilling I fell asleep. Bravo."* →
  sarcasm=1; sentiment=negative.
- **Not sarcasm (hyperbole):** *"This is the worst movie ever made, and I've
  seen everything."* → sarcasm=0; sentiment=negative.
- **Mixed:** *"The cinematography alone is worth the ticket, every frame a
  painting — but the script is a disaster, and the last act collapses
  entirely."* → mixed=1.
- **Not mixed:** *"Great film. Only downside is the run time, but honestly I
  didn't notice it."* → mixed=0.
- **Ambiguous:** *"I've watched it twice now and I still can't tell you whether
  I liked it. Maybe it's brilliant. Maybe it's nothing."* → ambiguous=1.
- **Not ambiguous (decided mixed):** *"Beautiful but boring. I admire it and I
  won't rewatch it. Verdict: skip it."* → mixed=1, ambiguous=0.
- **Complex:** three sentences totalling ~90 words with subordinate clauses,
  abstract terms ("ontological emptiness of late-capitalist nostalgia"), essay
  structure. → complex=1.
- **Not complex:** *"Loved it. Funny from start to finish. Best comedy this
  year."* (long reviews excluded per §1.4) → straightforward.
- **Multiple:** a sarcastic, essay-length pan with one begrudging compliment
  and a hedged verdict → sarcasm=1, mixed=1, ambiguous=1, complex=1.

## 4. Rules for reviews in multiple categories

1. **Multi-label, not forced-choice.** The four flags are independent binary
   judgments; a review may carry any combination. **[Decision]** This preserves
   information and lets us test each category's association with errors
   separately, as plan.md §4 requires.
2. **`straightforward` is derived** (= 1 iff all four flags are 0), so it is
   mutually exclusive with every other category by construction.
   **[Decision]**
3. **Sentiment tie-break (needed when ambiguous = 1):** (a) if the reviewer
   states a recommendation ("watch it" / "skip it"), follow it; (b) else use
   the direction of the *final* evaluative statement about the movie; (c) else
   use the direction of the review's first evaluative statement; (d) if still
   50/50, choose the side the reviewer's tone leans toward and set
   `uncertainty_sentiment` = 2. The applied tie-break step is recorded in
   `notes`. **[Decision]**
4. **No primary-category priority.** Per the approved design, no forced-choice
   primary category is imposed for the main analysis: accuracy comparisons use
   each binary flag (and the derived `straightforward` reference class)
   separately. **[Decision]**

## 5. Annotation format

One row per review, in a CSV (e.g., `data/annotations.csv` — location to be
confirmed before the sampling step; *not* in `data/raw/`):

| Field | Type | Values |
|---|---|---|
| `review_id` | str | ID assigned by the blind sampling script (random, not the dataset row index, so it reveals nothing) |
| `annotator` | str | Your initials |
| `sentiment` | categorical | `positive` / `negative` (forced choice; binary like the dataset) |
| `sarcasm` | 0/1 | |
| `mixed` | 0/1 | |
| `ambiguous` | 0/1 | |
| `complex` | 0/1 | |
| `straightforward` | 0/1 | derived = 1 iff all four flags are 0 (computed by script, not typed) |
| `uncertainty_sentiment` | 0–2 | 0 = certain, 1 = somewhat unsure, 2 = essentially guessed (tie-break used) |
| `uncertainty_categories` | 0–2 | same scale, for the four flags as a group |
| `notes` | free text | tie-break step used, quotes, anything unusual |
| `annotation_timestamp` | ISO timestamp | |

Every derived field (`straightforward`) is computed by a script that prints row
counts, per AGENTS.md.

## 6. Blind annotation procedure for 200 reviews

1. **Sampling [Decision]:** 200 **unique** texts drawn only from the **test
   split** (the split the model will be evaluated on). A script: fixes a random
   seed (logged in the script and its output), shuffles, takes 200 unique
   texts, and writes a blind annotation file containing **only `review_id` +
   `text`** (with `<br />` → line break). The mapping `review_id` → dataset row
   is saved in a separate file the annotator does not open during annotation.
   Row counts printed at every step per AGENTS.md.
2. **Why test split [Decision]:** the error analysis compares model predictions
   with dataset labels; annotating the same split the model is evaluated on
   avoids cross-split confounds.
3. **Blindness safeguards:** the annotator never opens `data/raw/` files with
   labels, never opens the ID-mapping file, and does not sort or group reviews
   by any characteristic before/during annotation (order is the fixed
   shuffle). The dataset label meaning (0/1) is irrelevant during annotation.
4. **After annotation is complete:** the mapping is joined back only for the
   evaluation step, and the join script prints row counts and verifies a
   200/200 match.
5. **Duplicate note [Decision]:** sampled texts are unique, but a sampled text
   may still occur twice in the full test split (199 duplicate rows exist
   there — data/README.md §4). The sampling script will record whether a
   sampled text has a twin, since its label twin is by definition also in the
   split.

## 7. Consistency safeguards

- **Pilot:** annotate 10 reviews first, then re-read the codebook and only then
  continue. **[Decision]**
- **Intra-annotator reliability [Decision]:** after finishing the 200,
  re-annotate ~20 randomly chosen reviews (same seed-logged selection) at least
  one day later, without consulting the first pass; agreement rates go to
  `results/` via a script. This is the minimum consistency check available for
  a single annotator.
- **Codebook freezes at v1.0** once annotation starts; any revision is logged
  with a version bump and applied only to reviews not yet annotated.
  **[Decision]**

## 8. How this tests the plan.md hypothesis

- `straightforward` = reference class; accuracy (model vs dataset label) is
  compared for each binary difficulty flag vs. `straightforward` — with no
  imposed primary-category priority, per the approved design.
- plan.md §4's falsification condition — sarcastic/mixed/ambiguous/complex
  reviews performing *as well or better* — is directly computable from these
  labels.
- Per AGENTS.md: no causal claims from this observational comparison;
  differences will be described as associations. Any uncertainty in
  annotations is carried through as the `uncertainty_*` fields, and analyses
  will report sensitivity to uncertainty (e.g., excluding
  `uncertainty_sentiment` = 2 rows).

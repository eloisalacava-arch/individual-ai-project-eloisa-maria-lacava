# data/README.md — IMDb Sentiment Dataset

Documentation of the raw input data for this project (sentiment classification
error analysis; see `plan.md`).

Status labels used throughout:

- **[Verified]** — independently verifiable from a file in `results/`; the
  generating script is named.
- **[Agent calculation]** — derived by the analysis agent from verified
  numbers, but not printed in any `results/` file.
- **[External]** — taken from dataset documentation (Hugging Face dataset card
  for `stanfordnlp/imdb`; Maas et al. 2011); not verifiable from local files.

`data/raw/` is read-only (AGENTS.md). No filtering, merging, or dropping of rows
has been performed on this data; no duplicates have been removed.

---

## 1. Dataset name, source, and purpose

- **Name:** IMDb Large Movie Review Dataset (Hugging Face version:
  `stanfordnlp/imdb`). **[External]**
- **Source:** Downloaded from Hugging Face (`stanfordnlp/imdb`) and stored
  locally as Parquet in `data/raw/`:
  - `train-00000-of-00001.parquet`
  - `test-00000-of-00001.parquet`
- **Original provenance:** Movie reviews scraped from the IMDb website;
  sentiment labels assigned by the dataset authors (Maas et al., 2011,
  *Learning Word Vectors for Sentiment Analysis*). **[External]**
- **Purpose in this project:** Evaluate how accurately an AI model classifies
  movie reviews as positive or negative, and identify which review
  characteristics are associated with classification errors (research question
  in `plan.md`).
- The raw files in `data/raw/` are never modified; all derived data must be
  written to other directories. **[Project rule, AGENTS.md]**

## 2. Column descriptions

Both files contain the same two columns. **[Verified —
results/04_verify_data_quality.txt, src/04_verify_data_quality.py]**

| Column | Data type | Meaning | Possible values | Units |
|---|---|---|---|---|
| `text` | `str` | Full plain-text body of one movie review in English. Contains IMDb web-scraping artifacts (see §4 and §5). | Free text. Length 52–13,704 characters (train), 32–12,988 (test); median 979 chars (train), 962 (test). **[Verified — results/04_verify_data_quality.txt, src/04_verify_data_quality.py]** | characters |
| `label` | `int64` | Binary sentiment label assigned by the dataset authors. | Observed values: `0` and `1` only, in both splits. **[Verified]** The intended meaning is `0` = negative, `1` = positive, per the dataset card. **[External]** | none (binary code) |

## 3. Number of observations

| Split | Rows | Negative (`label=0`) | Positive (`label=1`) | Status |
|---|---|---|---|---|
| train | 25,000 | 12,500 | 12,500 | **[Verified — results/01_verify_labels.txt, src/01_verify_labels.py; and results/04_verify_data_quality.txt, src/04_verify_data_quality.py]** |
| test | 25,000 | 12,500 | 12,500 | **[Verified — results/04_verify_data_quality.txt, src/04_verify_data_quality.py]** |
| **Total** | 50,000 | 25,000 | 25,000 | **[Agent calculation — sum of the two verified row counts]** |

The advertised total of 50,000 labelled reviews matches the observed row
counts. **[External — dataset card; consistent with the verified counts above]**

## 4. Missing values, duplicates, and data-quality issues

### Missing values

- No null values in `text` or `label` in either split, and no empty or
  whitespace-only texts. **[Verified — results/04_verify_data_quality.txt,
  src/04_verify_data_quality.py]**

### Duplicates (identical `text`)

- **train:** 96 duplicate rows (copies beyond the first occurrence) across 92
  duplicated-text groups → 24,904 unique texts. **[Verified —
  results/02_verify_duplicates.txt, src/02_verify_duplicates.py; and
  results/04_verify_data_quality.txt, src/04_verify_data_quality.py]**
- **test:** 199 duplicate rows across 191 duplicated-text groups → 24,801
  unique texts. **[Verified — results/04_verify_data_quality.txt,
  src/04_verify_data_quality.py]**
- No duplicated text carries conflicting labels in either split: all duplicated
  texts agree on their label. **[Verified — results/04_verify_data_quality.txt,
  src/04_verify_data_quality.py]**
- With 96 extra copies in 92 groups, at least 4 groups must contain 3 or more
  identical copies (if all groups had exactly 2 copies, there would be 92 extra
  copies, not 96). **[Agent calculation — derived from the verified counts
  above]**

### Train–test overlap

- 123 review texts appear in both the train and test splits. **[Verified —
  results/03_verify_overlap.txt, src/03_verify_overlap.py; and
  results/04_verify_data_quality.txt, src/04_verify_data_quality.py]**
- All 123 shared texts carry the same label in both splits. **[Verified —
  results/04_verify_data_quality.txt, src/04_verify_data_quality.py]**

### Other quality issues

- Web-scraping artifacts: 14,665 train texts and 14,535 test texts contain the
  HTML break tag `<br />`; 1,390 train and 1,392 test texts contain the `&`
  character (HTML entities such as `&quot;`). **[Verified —
  results/04_verify_data_quality.txt, src/04_verify_data_quality.py]**

## 5. Important limitations for sentiment classification

1. **Binary labels only — no neutral class.** Reviews with middle-ground
   ratings were excluded when the dataset was constructed (typically ratings
   ≤ 4 stars → negative, ≥ 7 stars → positive). **[External — Maas et al.
   2011]** This matters for our hypothesis in `plan.md`: mixed or ambiguous
   opinions are underrepresented by design, so classification errors on such
   reviews may be rarer in this data than in the real world.
2. **Label provenance is unverified.** Labels were assigned by the dataset
   authors from star ratings, not by reading the review text; a
   "negative"-labelled review can contain praise. We treat labels as ground
   truth for evaluation, but they cannot be independently verified from the
   local files. **[External]** No duplicate text has a conflicting label
   locally, which is consistent with consistent labelling but does not prove
   label correctness. **[Verified — results/04_verify_data_quality.txt]**
3. **Duplicates and train–test overlap can inflate accuracy.** 96 duplicate
   rows in train, 199 in test, and 123 texts shared across splits mean the
   rows are not fully independent observations. **[Verified —
   results/02_verify_duplicates.txt, results/03_verify_overlap.txt,
   results/04_verify_data_quality.txt]** Because all shared texts agree on
   their label across splits, this creates no label noise, but evaluation must
   account for the overlap before drawing conclusions. **[Verified +
   agent interpretation]**
4. **Web-scraping artifacts.** Roughly 59% of texts in each split contain
   `<br />` tags **[Agent calculation — computed from the verified artifact
   counts and row counts in results/04_verify_data_quality.txt]**, and some
   contain HTML entities. Preprocessing choices (whether to strip these)
   may affect measured error rates.
5. **Domain and coverage limits.** The data consists of English-language
   movie reviews collected up to the dataset's construction date (2011).
   **[External]** Findings do not generalize to other domains, languages,
   or recent movies.
6. **Self-selected, long reviews.** Reviews are volunteered by users and are
   fairly long (median 979 characters in train). **[Verified —
   results/04_verify_data_quality.txt]** Short or low-effort reviews are
   underrepresented.
7. **Minimal metadata.** Only `text` and `label` are present — no reviewer,
   movie, star-rating, or date information — so review characteristics used in
   the error analysis must be derived from the text itself.

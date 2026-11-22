# Publication-safe evidence and reports

This directory contains the public copies of the project's reports and supporting evidence. The original files under `runs/`, `reports/`, `data/` and `comparison/` were not modified.

## What is here

| Path | Contents |
|---|---|
| `reports/imdb_case_study/study_report.md` / `.pdf` | IMDb statistical report, all tables, figures, conclusions and limitations |
| `reports/imdb_case_study/qualitative_report.md` / `.pdf` | IMDb qualitative aggregate findings and limitations |
| `reports/phrasebank_75agree/study_report.md` / `.pdf` | Financial PhraseBank statistical report, all tables, figures, conclusions and limitations |
| `reports/phrasebank_75agree/qualitative_report.md` / `.pdf` | PhraseBank qualitative aggregate findings and limitations |
| `reports/comparison/report.md` / `.pdf` | Cross-dataset comparison report |
| `provenance/` | Sanitized manifests, audit logs, evaluation logs and run metadata |
| `qualitative/` | Aggregate qualitative tables and settings; no example rows |
| `context/` | Earlier group-project context, with annotator initials redacted |
| `data/annotations_public.csv` | Annotation table without the `annotator` column |
| `results/README.md` | Pointer to the exact result tables kept at their original `runs/` paths |
| `RESULTS_ALLOWLIST.md` | Exact original result tables and hashes selected for statistical reproduction |
| `MANIFEST.md` | SHA-256 hashes for every original and published copy, with the redaction applied |

## Report sanitization

The original study and qualitative reports contain an **Illustrative examples** section with short verbatim quotations from reviews and financial sentences. Those quotations are not redistributed here.

The public copies:

- remove the illustrative-example rows and their `evidence:` quotations;
- keep every statistical table, figure, numeric result, interpretation, conclusion and limitations section;
- keep the aggregate qualitative tables, control contrast and explicit hypothesis caveats;
- replace the removed examples with a short note explaining the omission.

The Markdown and PDF copies are generated from the original report structure. All remaining lines are unchanged from the original after the example block is removed. The manifest records the original and published SHA-256 values.

## Provenance redactions

Publication copies of JSON and text metadata replace local absolute paths:

- the authoring project root → `<project-root>`
- the authoring home directory → `<home>`

`decisions.json` also replaces the dataset text examples with `<redacted: source text>`. Counts, labels, hashes and timestamps are unchanged. The `annotator` column is removed from `annotations_public.csv`, and annotator initials are redacted from the group-context verification copy. Files that needed no redaction are marked as byte-identical copies in `MANIFEST.md`.

## What the repository can and cannot reproduce

**It can reproduce the statistics without a model call.**

- `uv run pytest tests/test_reproduce_case_studies.py -rs` recomputes accuracy and Cohen's kappa from the tracked frozen predictions and sealed benchmark labels.
- `uv run python scripts/09_report.py --all` regenerates statistical reports from the tracked result tables.
- `uv run python scripts/09_report.py --comparison` regenerates the comparison report from the tracked comparison tables and figures.

**It cannot reconstruct the original classification run.**

Reconstruction would require the excluded observation text (`blind/blind.csv`), the source datasets, a configured provider, and paid model calls with a new pilot authorization. The public repository is a statistical verification package, not a fully self-contained classification rerun. This is a deliberate licence and privacy choice.

The static reports under `reports/` preserve the qualitative aggregate findings without source quotations. A fresh clone's `09_report.py --all` run will produce statistical reports without the qualitative section, because the raw qualitative explanations are not published.

## Known issue

The IMDb `t08_disagreements.csv` severity column labels the six binary disagreements `adjacent`, while `t07_ordinal_errors.csv` and the cross-dataset severity table correctly label them `opposite`. No metric or count changes. The original CSV has not been altered. See the Error Analysis, §8. The correct `t07` and comparison tables are the authoritative severity summary.

## Licence

Financial PhraseBank is CC BY-NC-SA 3.0. Its sentence text is not redistributed here; `scripts/fetch_dataset.py` retrieves it into the gitignored `inputs/` directory. The raw IMDb archive and the 200-review text input are also excluded. See the repository root README and `data/README.md`.
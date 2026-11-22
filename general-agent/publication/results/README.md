# Result tables for statistical reproduction

The exact original result tables are kept at their original paths:

- `general-agent/runs/imdb_case_study/results/`
- `general-agent/runs/phrasebank_75agree/results/`

They are not copied here. `../RESULTS_ALLOWLIST.md` lists every selected file with its SHA-256 and the audit result. The reproduction test reads:

- `runs/<run_id>/results/frozen_predictions.csv`
- `runs/<run_id>/sealed/benchmark.csv`
- `runs/<run_id>/results/t01`–`t16`

Keeping the tables in place avoids duplicate evidence and lets `tests/test_reproduce_case_studies.py` run without changing the test's fixed `runs/` path. The original tables were not modified.
# AGENTS.md — rules for `general-agent/`

Scoped rules for this folder. The parent project's `AGENTS.md` still applies to
the parent repository; where they overlap, the stricter rule wins.

1. **No label leakage.** The classifier reads only the blind input file
   (`id`, `text`). No benchmark value, count, distribution, or class set may
   enter a prompt except the declared class names that the task itself must
   state. Evaluation runs after predictions are frozen.
2. **Print row counts after every filter, merge, or drop, with the reason.**
   The counts also go into `manifest.json → row_accounting`.
3. **Every number in a write-up comes from a CSV or JSON under
   `runs/<run_id>/results/`.** The report generator reads files, never console
   output or prose.
4. **Fix seeds.** Every random operation (sampling, bootstrap, subset
   selection) uses `run.seed`, recorded in the manifest.
5. **Log every model call**: full prompt, raw output, model string, attempt
   number, timestamp, and outcome. Never overwrite a log; append and freeze.
6. **Never impute a missing prediction.** Report it as a failure under a stated
   policy.
7. **`not_estimable` is a valid result.** Degenerate statistics report a reason
   instead of a number.
8. **No causal claims.** Comparisons are associational; state sample sizes and
   minimum detectable differences before saying "no difference".
9. **Inputs are read-only.** Never modify a configured dataset or benchmark.
   Derived files go under `runs/` or `inputs/`.
10. **Credentials stay in the environment.** Configs name an environment
    variable (`api_key_env`), never hold a key. `.env` is gitignored.
11. **Say "I am not sure"** when the evidence does not support a statement, and
    label external claims `[External]`.
12. **Ask before changing `SPEC.md`.** A change is a version bump plus a
    changelog entry.
13. **Do not add dependencies for convenience.** The runtime set is small on
    purpose so the folder installs on a student's machine.
14. **A model call requires an explicit go-ahead** until the budget caps and
    the pilot gate are approved.
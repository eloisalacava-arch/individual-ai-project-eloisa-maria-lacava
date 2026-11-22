"""Evaluation: freeze predictions, join the reference, compute metrics.

Stage 05 opens the benchmark only after the predictions are hashed and copied
aside. Importing this package from the classifier is forbidden and enforced by
an import-boundary test.
"""
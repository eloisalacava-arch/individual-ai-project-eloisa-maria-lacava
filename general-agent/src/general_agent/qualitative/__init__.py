"""Stage 08 - qualitative disagreement analysis.

An evaluation-side stage: it runs only after predictions are frozen, reads the
human reference and the disagreement set, asks a model for a structured
*hypothesis* about each disagreement, aggregates recurring reasons
deterministically, and composes a narrative report that embeds the statistical
report.

Nothing here can change a prediction, and no classifier module imports it.
"""

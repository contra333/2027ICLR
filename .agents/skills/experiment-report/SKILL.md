---
name: experiment-report
description: Use when summarizing experiment logs, CSV/JSON metrics, result manifests, or professor-facing experiment reports.
---

# Experiment Report

## Required behavior

- Separate confirmed metrics from interpretation.
- State exact files read.
- Do not invent missing seeds, averages, or failed-run explanations.

## Steps

1. Read relevant `results/manifests/*.json`.
2. Read only the metrics/log files needed for the question.
3. Extract metrics without changing their values.
4. Separate confirmed results, interpretation, and risks.
5. Write a concise report under `reports/` only if the user asks for a file.

## Output format

1. Purpose
2. Files checked
3. Confirmed results
4. Interpretation
5. Missing evidence or risks
6. Next actions

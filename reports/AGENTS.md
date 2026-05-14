# reports/AGENTS.md

## Role

`reports/` contains experiment summaries, professor-facing reports, analysis notes, and paper interpretation drafts.

## Daily report workflow

- For daily research records or paired Markdown/HTML reports, read `reports/DAILY_REPORT_WORKFLOW.md` first.
- Write or update the Markdown work log before generating the HTML report.
- Treat the Markdown as the source of truth; the HTML should not introduce new claims, metrics, or results absent from the Markdown.
- Use `reports/templates/daily_work_log.md` and `reports/templates/daily_report.html` as starting points when creating a new daily report pair.
- For date-specific context, use the relevant daily Markdown report in `reports/` alongside `AI_CONTEXT.md`.
- `AI_CONTEXT.md` is the compact project restart note; daily reports are the chronological record of what happened each date.

## Reporting rules

- Separate confirmed results, interpretation, and next hypotheses.
- Before interpreting evaluator outputs or naming metrics, read `reports/METRIC_DEFINITIONS.md` and follow its score direction and naming rules.
- Do not report `nc4` as label accuracy. Distinguish `ncc_accuracy` from strict `nc4_agreement`.
- Treat `neco_lite` as a component diagnostic unless a fixed aggregate formula is documented; do not call it official NECO before `neco_full` is implemented against the paper or official code.
- Do not invent missing metrics or smooth over failed runs.
- State exact files read for any result summary.
- When writing paper-facing text, preserve the source boundary between NeurIPS results, ICLR hypotheses, external-paper claims, and new experiment results.
- If a report uses a figure or table, name the generating script or input result files when available.

## Suggested report structure

1. Purpose
2. Files checked
3. Confirmed metrics
4. Interpretation
5. Risks or missing evidence
6. Next actions

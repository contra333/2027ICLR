# Daily Research Report Workflow

## Purpose

Daily research work should leave two durable artifacts:

1. A Markdown work log that is easy for AI systems to read later.
2. An HTML report that is easy for a human reader to scan, print, and review.

The Markdown file is the source of truth for the day's context. The HTML file is a presentation artifact derived from that Markdown. Do not write the HTML first, and do not add new research claims directly in HTML that are absent from the Markdown source.

## File Pair

Use this filename pattern by default:

| Artifact | Pattern | Role |
|---|---|---|
| Markdown work log | `reports/MMDD_<short-topic>.md` | AI-readable record of what happened, what was read, what was decided, and what remains uncertain. |
| HTML report | `reports/MMDD_<short-topic>.html` | Human-readable report with compact sections, tables, status labels, and clear evidence boundaries. |

Keep the pair adjacent in `reports/`. Do not create a new top-level folder for daily reports unless explicitly requested.

## Writing Order

1. Read the smallest relevant context first: `AI_CONTEXT.md`, `reports/AGENTS.md`, the current daily Markdown if it exists, and any files actually being summarized.
2. Write or update the Markdown work log first.
3. Check that the Markdown separates confirmed facts, interpretation, hypotheses, risks, and missing evidence.
4. Generate the HTML report from the Markdown content.
5. Verify that the HTML section order matches the Markdown section order.
6. Record any format feedback as a rule to apply to the next daily report.

## AI Context Usage

`AI_CONTEXT.md` gives the compact project-level restart context. Daily report pairs give date-by-date operational context. Future AI sessions should use both:

- read `AI_CONTEXT.md` for the current thesis, evidence boundaries, repo status, and next major actions
- read relevant `reports/MMDD_<topic>.md` files to understand what happened on a specific date
- treat daily Markdown reports as the source for day-level work history, decisions, feedback, and unresolved follow-ups
- use HTML reports for human-facing review and quick visual scanning, not as the source of new research claims

## Markdown Work Log Structure

Use `reports/templates/daily_work_log.md` as the starting point.

The Markdown should preserve context for future AI sessions. It should include:

- date, project, and short status
- the day's goal
- files or sources actually checked
- work completed
- decisions made
- confirmed facts, interpretation, hypotheses, risks, and missing evidence
- next actions
- feedback received about report structure or tone

Avoid polished-only prose. The Markdown should remain explicit enough that an AI system can reconstruct the day's context without guessing.

## HTML Report Structure

Use `reports/templates/daily_report.html` as the starting point.

The HTML should be a readable report, not a slide deck. Follow the report design system in `PRODUCT.md` and `DESIGN.md`:

- compact header
- numbered sections
- status labels with text, not color alone
- tables for evidence and next actions
- no hero section, decorative gradients, slide-like paging, or forced viewport-height sections
- no `scroll-snap`, `min-height: vh`, or `class="slide"` pattern

The HTML may compress wording for readability, but it must not introduce claims, metrics, or results that are not present in the Markdown source.

## Feedback Loop

Daily report format should improve over time.

When feedback arrives, convert it into a reusable rule:

| Feedback type | How to record it |
|---|---|
| Tone | Add a tone rule, such as formal Korean endings or avoiding direct reader labels. |
| Structure | Adjust section order or table columns in the next Markdown template use. |
| Design | Adjust the HTML template only when the change applies beyond one report. |
| Evidence boundary | Update the Markdown checklist so future reports separate facts, interpretation, hypotheses, risks, and missing evidence. |

Do not silently generalize one-off content feedback into a permanent rule unless it clearly applies to future reports.

## Current Format Rules

- Use formal written Korean for report prose.
- Do not address the reader with role-specific labels in document text.
- Keep technical terms such as optimizer names, detector names, metric identifiers, run IDs, and file paths exact.
- Preserve evidence boundaries. Do not write smoke validation, literature evidence, and full-scale experiment evidence as one proof layer.
- For file transformations or report generation, state the exact source files used.
- If no new experiment result exists, say so directly instead of implying progress evidence.

## Verification Checklist

Before calling a daily report pair complete:

- The Markdown exists and is readable.
- The HTML exists and parses as HTML.
- The HTML section order follows the Markdown.
- The HTML does not contain `scroll-snap`, `min-height: vh`, `Slide-like`, or `class="slide"`.
- Status labels include text such as `확정`, `해석`, `가설`, `위험`, or `누락`.
- Claims are traceable to files read, prior summaries, manifests, or clearly marked interpretation.

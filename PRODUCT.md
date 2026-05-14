# Product

## Scope

This document is presentation context for impeccable-generated report artifacts only. It is not a research source, metric source, experiment interpretation source, or paper-claim authority.

It applies only to HTML reports, daily research records, progress reports, and related document outputs whose job is to make the current research state easier to read. It must not influence research hypotheses, experiment analysis, metric definitions, paper claims, or decisions about what the evidence proves.

When a report needs research truth, use the authoritative sources instead: `AI_CONTEXT.md`, `소스/INDEX.md`, `reports/METRIC_DEFINITIONS.md`, `results/manifests/*.json`, raw/server result artifacts, and the relevant `AGENTS.md` files.

## Register

product

## Primary Audience

The report reader values clear logical structure, itemized reasoning, good readability, and fast comprehension. A strong report should let a non-specialist understand the research flow within about three minutes without making the document feel thin or casual.

The document should not address the reader with role-specific labels. It should make the reasoning inspectable through structure: section order, status labels, tables, evidence boundaries, and concise explanatory text.

## Product Purpose

The product is a reporting system for the 2027 ICLR research project. It turns literature notes, experiment manifests, code/config status, server-result summaries, and daily progress into consistent reports with explicit evidence boundaries.

Success means the report makes the current research situation clear without pretending to settle the research itself. A reader should be able to see:

- what question the project is currently asking
- how the logical chain is supposed to work
- what evidence has already been secured
- what changed in the latest work session
- what remains uncertain or missing
- what concrete work follows next

## Quality Principles

1. Logic first. Reports should flow from `problem/question -> logic structure -> secured evidence -> latest work -> remaining uncertainty -> next work`.
2. Evidence levels must be visible. Every claim should be framed as one of `확정`, `해석`, `가설`, `위험`, or `누락`.
3. Do not flatten evidence tiers. Smoke results, literature evidence, full-scale experiment evidence, and paper-ready claims must never be written as if they have the same authority.
4. Use itemization without making the document look bare. Prefer short paragraphs, tables, labeled callouts, and compact bullets over long prose.
5. Every section should answer "why this matters" through one short interpretation sentence or a clear table column.
6. Avoid plausible but vague phrasing. Prefer sentences that expose the source, status, implication, and next check.
7. Make specialist terms readable. Keep English research terms and metric names intact, but provide a short Korean reading guide when the term affects the main logic.

## Standard Report Structure

1. 연구 배경과 현재 질문
2. 논리 구조
3. 현재까지 확보된 근거
4. 오늘 진행한 작업
5. 현재 해석과 남은 불확실성
6. 다음 작업 계획
7. 용어

Each report may omit sections that are genuinely irrelevant, but it should preserve this order when the full research briefing structure is needed.

## Required Report Behaviors

- Start with the research question and context, not a forced one-line conclusion.
- Do not add a self-describing preface under the title. The report should not explain that it is a report; it should immediately show the research state through structured content.
- Show the logic chain before presenting detailed daily work.
- Use short, direct section titles such as `용어` or `Notation`; avoid manual-like headings such as `읽는 법`.
- Do not add a separate reading-path card unless it matches every visible section exactly and adds clear value beyond numbered headings.
- Separate `confirmed fact`, `interpretation`, `hypothesis`, `risk`, and `missing evidence`.
- Attach claims to files, manifests, reports, or source notes when possible.
- State when raw results are absent from the local checkout and a report is relying on manifests or prior summaries.
- Keep the tone scholarly, direct, and operational. The report should feel carefully prepared, not promotional.
- Use Korean as the main language, while preserving English paper terms, optimizer names, detector names, and metric identifiers.
- Keep tables visually stable by aligning cells according to column meaning: headers centered, row-label columns centered, status-pill columns centered, short categorical/numeric columns centered, and prose/explanation columns left-aligned. Korean/English terms should not break awkwardly inside words.

## Anti-References

- Marketing landing pages, hero sections, background images, and decorative product storytelling.
- Dashboard decoration that looks impressive but hides the research logic.
- Generic AI-style summaries that announce a conclusion before showing the structure.
- Self-explaining title prefaces that describe the report instead of entering the research structure.
- Navigation cards or reading paths that do not match the actual section order.
- Manual-like phrases such as `읽는 법`, `이 보고서는`, or `아래 용어는` when a direct heading or table is enough.
- Role-addressed judgment-point headings that make the document sound like a generated briefing.
- Treating 2-epoch smoke validation as paper-level evidence.
- Treating external literature, local smoke validation, and full-scale experimental evidence as one blended proof layer.
- Rewriting metric definitions, experimental meaning, or paper claims from presentation context.

## Accessibility & Inclusion

Reports should be readable in a browser, printable when needed, and understandable without relying on color alone. Status labels must include text, not only color. Tables should use clear headers, enough spacing, and compact but legible type.

Use plain Korean explanations for difficult terms, but do not erase the English technical vocabulary needed for research traceability.

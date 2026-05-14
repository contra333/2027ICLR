---
name: "2027 ICLR Research Report System"
description: "A restrained HTML reporting system where research logic, evidence level, and next work are visible at a glance."
colors:
  report-ink: "#17202a"
  report-muted: "#5f6c7b"
  report-bg: "#f5f7fa"
  report-surface: "#fbfcfe"
  report-panel: "#ffffff"
  report-line: "#d9e0e8"
  navy: "#152842"
  navy-deep: "#132033"
  blue: "#245b9f"
  blue-soft: "#e8f1fb"
  green: "#136f4b"
  green-soft: "#e8f6ef"
  purple: "#694f9f"
  purple-soft: "#f0ebfa"
  amber: "#936000"
  amber-soft: "#fdf1d3"
  red: "#9f2d3a"
  red-soft: "#fdebed"
  code-bg: "#eef3f8"
typography:
  display:
    fontFamily: "Segoe UI, Noto Sans KR, Arial, sans-serif"
    fontSize: "28px"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "normal"
  headline:
    fontFamily: "Segoe UI, Noto Sans KR, Arial, sans-serif"
    fontSize: "22px"
    fontWeight: 750
    lineHeight: 1.35
    letterSpacing: "normal"
  title:
    fontFamily: "Segoe UI, Noto Sans KR, Arial, sans-serif"
    fontSize: "17px"
    fontWeight: 750
    lineHeight: 1.4
    letterSpacing: "normal"
  body:
    fontFamily: "Segoe UI, Noto Sans KR, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.68
    letterSpacing: "normal"
  table:
    fontFamily: "Segoe UI, Noto Sans KR, Arial, sans-serif"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: "Segoe UI, Noto Sans KR, Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "normal"
  mono:
    fontFamily: "Consolas, Cascadia Mono, monospace"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "normal"
rounded:
  sm: "4px"
  md: "8px"
  lg: "12px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  report-section:
    backgroundColor: "{colors.report-panel}"
    textColor: "{colors.report-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "24px"
  report-header:
    backgroundColor: "{colors.navy}"
    textColor: "{colors.report-surface}"
    typography: "{typography.display}"
    rounded: "{rounded.md}"
    padding: "28px 32px"
  status-confirmed:
    backgroundColor: "{colors.green-soft}"
    textColor: "{colors.green}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "3px 8px"
  status-interpretation:
    backgroundColor: "{colors.purple-soft}"
    textColor: "{colors.purple}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "3px 8px"
  status-risk:
    backgroundColor: "{colors.amber-soft}"
    textColor: "{colors.amber}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "3px 8px"
  status-missing:
    backgroundColor: "{colors.red-soft}"
    textColor: "{colors.red}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "3px 8px"
  logic-flow:
    backgroundColor: "{colors.blue-soft}"
    textColor: "{colors.navy}"
    typography: "{typography.title}"
    rounded: "{rounded.md}"
    padding: "16px"
---

# Design System: 2027 ICLR Research Reports

## Overview

The creative north star is "visible logic, polished research briefing." The report should look prepared, rigorous, and easy to inspect. Its polish comes from hierarchy, tables, status labels, and evidence boundaries rather than decoration.

This is a product-style document surface. Design serves the report task: helping the reader understand the research state quickly while preserving uncertainty and source boundaries. It should feel like a serious internal research briefing, not a marketing page, slide deck, or generic AI-generated summary.

Use the full report structure when a research update needs context: research background and current question, logic structure, secured evidence, latest work, interpretation and uncertainty, next work, and terms. Shorter reports may compress sections, but the logical order should remain intact.

Key characteristics:

- high information density with generous enough spacing to remain calm
- numbered sections that make the research flow visible
- status labels that make evidence level impossible to miss
- tables that connect claim, source, status, implication, and next check
- Korean-first explanations with English research terms preserved for traceability
- no separate reading-path cards when numbered sections already provide the navigation

## Colors

The palette is restrained and semantic. Navy carries document authority, blue carries information and progress, and green, purple, amber, and red are reserved for evidence status.

Primary colors:

- Deep Research Navy (`#152842`): page header, main title areas, and the visual anchor for the report.
- Report Blue (`#245b9f`): information, progress, links, and logic-flow emphasis.

Status colors:

- Confirmed Green (`#136f4b` on `#e8f6ef`): facts that are directly confirmed by a source, manifest, metric file, or checked report.
- Interpretation Purple (`#694f9f` on `#f0ebfa`): analysis, reading, or hypothesis that is useful but not yet paper-level evidence.
- Risk Amber (`#936000` on `#fff5dc`): risks, caveats, fragile assumptions, or work that needs careful follow-up.
- Missing Red (`#9f2d3a` on `#fdebed`): absent raw results, missing evidence, unresolved checks, or incomplete artifacts.

Neutral colors:

- Ink (`#17202a`): body text and table content.
- Muted Ink (`#5f6c7b`): metadata, secondary labels, and explanatory notes.
- Paper Background (`#f5f7fa`): page background.
- Panel White (`#ffffff`): report sections and tables.
- Rule Line (`#d9e0e8`): borders, table dividers, and low-emphasis separators.

Named rule: semantic color only. Do not use status colors as decoration. A colored element must communicate evidence level, progress, risk, or source role.

## Typography

Use `Segoe UI`, `Noto Sans KR`, `Arial`, and `sans-serif` for report text. Use `Consolas` or `Cascadia Mono` only for paths, metric names, command fragments, run IDs, and exact identifiers.

The type system should feel precise and readable rather than expressive. Avoid oversized headings. Hierarchy should come from section numbering, weight, spacing, and table structure.

Hierarchy:

- Display: 28px, weight 800, line-height 1.25. Use only for the report title.
- Headline: 22px, weight 750, line-height 1.35. Use for major numbered sections.
- Title: 17px, weight 750, line-height 1.4. Use for subsection headings and logic-step labels.
- Body: 15px, line-height 1.68. Use for short explanatory paragraphs and bullets.
- Table: 13.5px, line-height 1.5. Use for evidence tables, progress tables, and glossary tables.
- Label: 12px, weight 700. Use for status pills, table labels, source tags, and small metadata.
- Mono: 13px. Use sparingly for exact machine-readable names.

For Korean report prose, do not constrain ordinary section paragraphs with narrow `ch`-based widths on desktop. Section introductions should align with the section content width, then rely on concise writing, tables, and callouts for readability.

For tables, align by column meaning rather than applying one body alignment everywhere. Header cells are centered. Row-label columns, status-pill columns, and short categorical or numeric columns are centered. Prose, explanation, source notes, and caveat columns are left-aligned. Apply `word-break: keep-all` and `overflow-wrap: break-word` so Korean labels and English research terms do not split letter-by-letter. If a short header or first-column label still breaks awkwardly, adjust column widths or shorten the label instead of accepting broken words.

Named rule: preserve terms. Keep optimizer names, detector names, NC component names, metric identifiers, run IDs, and file paths in their original English or exact form. Add a Korean reading guide when the term affects the main argument.

## Elevation

The system uses tonal layering with very light elevation. Surfaces should feel ordered, not floating. Section panels may use a subtle shadow only to separate report blocks from the page background.

Shadow vocabulary:

- Section lift: `0 8px 24px rgba(25, 42, 64, 0.06)`. Use only on major report sections.
- Header lift: `0 8px 26px rgba(30, 42, 60, 0.08)`. Use only for the top report header or a major title block.
- Table depth: no shadow. Tables use borders and header background instead.

Named rule: no nested-card look. Do not place a card inside another decorative card. If content needs grouping inside a section, use a table, a heading, a light background block, or a divider.

## Components

Numbered section block:

- Use for each major report section.
- Section titles should be numbered and written as research-document headings, such as `1. 연구 배경과 현재 질문`.
- Each section should open with one concise sentence that connects it to the previous section.
- Do not put a self-describing report explanation directly under the title. After the title, use metadata or structured status blocks, then enter the numbered research flow.
- Do not add a separate reading-path component when the numbered section headings already show the report order. If a navigation strip is used, it must match every visible major section exactly.

Logic flow block:

- Use to show the central chain, usually `Optimizer choice / WD coupling -> Feature geometry / NC regime -> Feature-based detector behavior -> Paper claim`.
- Prefer four compact cells or a single horizontal flow line.
- Keep the text technical but readable. Do not turn this into a decorative diagram.

Evidence boundary table:

- Required when a report discusses results or claims.
- Preferred columns: `항목`, `현재 상태`, `근거`, `아직 부족한 점`.
- Use status pills in the `현재 상태` column.
- Center-align the status column. Center-align the first row-label column when it contains compact labels; keep source/caveat prose columns left-aligned.

Today-progress table:

- Use for daily research records and progress reports.
- Preferred columns: `작업`, `확인한 내용`, `전체 논리에서의 의미`, `다음 확인`.
- Avoid writing a task list with no interpretation column.

Status pill:

- Use short text labels: `확정`, `해석`, `가설`, `위험`, `누락`.
- Each pill needs both color and text. Never rely on color alone.
- Use rounded 4px shape, compact padding, and bold 12px label text.
- Status pills inside tables must sit in a centered table cell, not drift against the left edge.

Risk or missing-evidence callout:

- Use for absent raw results, unverified metric files, seed-sweep gaps, full-training gaps, or claims that are not yet supported.
- Use amber for risks and red for missing evidence.
- Keep callout text direct: what is missing, why it matters, what check follows.

Terms table:

- Use near the end of a report.
- Preferred columns: `용어`, `짧은 설명`, `이 연구에서의 역할`.
- The section title should be short: `용어` for Korean reports or `Notation` when the content is mathematical notation. Avoid `읽는 법` unless the section is genuinely an instruction manual.
- The explanation should be understandable to a non-specialist without deleting the technical term.

File/source trace block:

- Use when the report depends on local files, manifests, or literature packages.
- Include exact file names or paths when useful.
- Make clear whether the report read raw results, manifests, source cards, or prior summaries.

## Do's and Don'ts

Do:

- Do make the logic order visible: current question, logic structure, evidence, latest work, uncertainty, next work, terms.
- Do use status pills to separate `확정`, `해석`, `가설`, `위험`, and `누락`.
- Do use tables for claim/source/status relationships rather than long paragraphs.
- Do center table header cells, row-label columns, status columns, and short categorical/numeric columns.
- Do keep prose, explanation, source-note, and caveat columns left-aligned.
- Do adjust column widths when short labels break awkwardly across lines.
- Do give each major section a short connective sentence so the report reads naturally.
- Do let Korean section prose use the full content width on desktop unless there is a deliberate side-by-side layout.
- Do keep the report polished through alignment, spacing, typography, and precise labels.
- Do state when smoke validation is only pipeline validation and not paper-level evidence.
- Do keep file paths, metric names, run IDs, optimizer names, and detector names exact.

Don't:

- Don't use a forced one-line conclusion, consulting-style summary title, or reader-addressed judgment-point heading.
- Don't put a title-level paragraph that starts by explaining what the report is doing.
- Don't make body prose look like an unused second column by applying narrow desktop `max-width` rules to ordinary section paragraphs.
- Don't use a reading-path card that omits sections or renames them differently from the real headings.
- Don't allow Korean labels or English technical terms to split letter-by-letter in table cells.
- Don't use manual-like headings or intro copy such as `읽는 법` or `아래 용어는` when a direct section title is enough.
- Don't use hero images, large decorative gradients, gradient text, glassmorphism, or marketing copy.
- Don't use side-stripe borders as colored accents. Use full borders, soft backgrounds, or status pills instead.
- Don't make every section an identical card grid.
- Don't hide missing raw results or unresolved checks.
- Don't write smoke results, literature evidence, and full-scale experimental evidence as if they have the same strength.
- Don't let this design document change research claims, experiment interpretation, or metric definitions.

# 소스/AGENTS.md

## Role of this folder

`소스/` is the evidence and source-material folder. It is not a scratch space.

Use it to store:

- original papers and project source documents,
- AI-readable derived text,
- source cards,
- manifests,
- source boundaries and evidence tiers.

## Source preservation

- Do not overwrite original PDFs, source archives, or imported raw materials.
- Create derived Markdown/text files instead of modifying originals.
- If extracting from a PDF, keep page-level or section-level anchors when possible.
- If a source is copied from another workspace, record the original path and checksum when available.

## Evidence policy

Separate every claim into one of these types:

- `confirmed`: backed by primary evidence or project result files.
- `interpretation`: reasoned reading of confirmed evidence.
- `hypothesis`: plausible but not yet verified by this project.
- `unsupported`: not supported by the available source.

For source claims, prefer primary evidence:

- original PDF,
- TeX/source text,
- official code/config,
- page-anchored extraction,
- result file or run manifest.

Treat summaries and assistant-written notes as navigation aids, not final evidence.

## Required source manifest fields

For new source packages, create or update a manifest with these fields:

- `source_id`
- `title`
- `original_path`
- `checksum`
- `derived_files`
- `evidence_tier`
- `unsupported_boundary`

Existing legacy manifests may use nested fields. When touching them, preserve existing data and add missing normalized fields only when useful.

## Current hard boundary

`Optimizer choice matters for the emergence of Neural Collapse` supports optimizer-induced changes in NC / representation geometry and the coupled versus decoupled weight-decay axis.

It is not direct evidence that Energy, Mahalanobis, kNN, DDU, or other downstream uncertainty/OOD detectors fail or succeed.

Use downstream detector claims only as project hypotheses unless supported by the NeurIPS 2026 paper or new experiment results.

## Ingest workflow

1. Preserve the original file.
2. Create a source card or guide.
3. Create AI-readable derived text with page/section anchors.
4. Add or update a manifest.
5. Update `INDEX.md`.
6. Report what was verified and what remains uncertain.

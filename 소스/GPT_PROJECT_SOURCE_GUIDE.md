# GPT Project Source Guide: arXiv 2602.16642v3

## Purpose

Use this folder when a GPT Project needs to apply the reference paper `Optimizer choice matters for the emergence of Neural Collapse` accurately in context.

The paper is useful as evidence for this axis:

- optimizer choice changes representation geometry / Neural Collapse emergence
- coupled vs decoupled weight decay is a meaningful intervention axis
- NC metrics can split, so geometry should not be reduced to a single score

Do not use this paper alone as direct evidence that a downstream uncertainty detector family fails. It does not directly evaluate Energy, Mahalanobis, kNN, DDU, or related uncertainty/OOD methods.

## Recommended Upload Set

Upload these files first:

1. `GPT_PROJECT_SOURCE_GUIDE.md`
2. `context_manifest.json`
3. `SRC-arxiv-2602.16642-v3.md`
4. `paper_source_tex_combined.md`
5. `paper_pdf_pages.md`

Add this original PDF when exact layout, equations, figures, or tables matter:

- `C:\Users\User\Desktop\Wiki\20_원본자료\논문\arxiv-2602.16642\versions\v3\paper.pdf`

Use `alphaxiv_fulltext.md` only as optional secondary context:

- `C:\Users\User\Desktop\Wiki\20_원본자료\논문\arxiv-2602.16642\versions\v3\alphaxiv_fulltext.md`

## Evidence Tiers

Primary evidence:

- `paper_source_tex_combined.md`: best source for exact paper claims, theorem statements, experiment settings, and limitations.
- `paper_pdf_pages.md`: primary PDF page context with page anchors.
- `paper.pdf`: original visual source of truth for page layout, equations, tables, and figures.

Project/Wiki interpretation:

- `SRC-arxiv-2602.16642-v3.md`: local source card that states how this paper should and should not be used in the Wiki project.

Secondary evidence:

- `alphaxiv_fulltext.md`: fast reading aid only. Treat claims from it as candidate until checked against TeX or PDF page context.

Usually do not upload:

- `arxiv_source.tar.gz`: archive container; GPT Projects may not use it well.
- `arxiv_abs.html`: metadata page, lower value once the source card and manifest are included.
- `derived/source_tex/*.sty`, `.bst`: LaTeX style files, not paper evidence.
- all figure PDFs by default: upload a specific figure file only when asking about that figure.

## Suggested GPT Project Instruction

When using this paper, treat `paper_source_tex_combined.md`, `paper_pdf_pages.md`, and `paper.pdf` as primary evidence. Treat `alphaxiv_fulltext.md` as secondary context only. Preserve the source boundary: this paper supports optimizer-induced changes in Neural Collapse / representation geometry, but it is not direct evidence for downstream uncertainty detector degradation. For exact values, equations, tables, or hyperparameters, cite the TeX source or PDF page context.

## Key Source Boundary

Supported by this paper:

- optimizer choice affects NC / representation geometry
- coupled weight decay is important for NC emergence in the studied settings
- AdamW / decoupled WD adaptive optimizers can show larger NC metrics and incomplete or partial collapse patterns
- the paper reports extensive experiments and theoretical analysis centered on NC0 and related NC metrics

Not supported directly by this paper:

- Energy score failure or success
- Mahalanobis degradation
- kNN, DDU, or other feature-density detector behavior
- causal proof that NC changes cause downstream uncertainty divergence

Use those as hypotheses or project-specific claims only when supported by separate experiments or sources.
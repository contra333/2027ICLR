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

1. `AI_README.md`
2. `manifest.json`
3. `SOURCE_CARD.md`
4. `paper_pdf_pages.md`
5. `GPT_PROJECT_SOURCE_GUIDE.md`

Add this original PDF when exact layout, equations, figures, or tables matter:

- `C:\Users\User\Desktop\2027ICLR\소스\paper.pdf`

Optional legacy context files in this folder:

- `SRC-arxiv-2602.16642-v3.md`
- `context_manifest.json`

## Evidence Tiers

Primary evidence:

- `paper_pdf_pages.md`: primary PDF page context with page anchors.
- `paper.pdf`: original visual source of truth for page layout, equations, tables, and figures.

Project interpretation:

- `SOURCE_CARD.md`: current source card for this repository.
- `SRC-arxiv-2602.16642-v3.md`: legacy source card that states how this paper should and should not be used.

Secondary evidence:

- `context_manifest.json`: legacy manifest/provenance context from the earlier source package.

Usually do not upload:

- all figure files by default: upload a specific figure file only when asking about that figure.
- source archives or LaTeX support files if they are later added; upload the combined paper text or targeted source section instead.

## Suggested GPT Project Instruction

When using this paper, treat `paper_pdf_pages.md` and `paper.pdf` as primary evidence. Preserve the source boundary: this paper supports optimizer-induced changes in Neural Collapse / representation geometry, but it is not direct evidence for downstream uncertainty detector degradation. For exact values, equations, tables, or hyperparameters, cite the PDF page context and check the original PDF layout when needed.

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

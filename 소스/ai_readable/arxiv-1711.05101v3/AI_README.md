# AI Reading Guide: Decoupled Weight Decay Regularization

## Identity

- Source ID: `arxiv-1711.05101v3`
- arXiv ID: `1711.05101v3`
- Title: Decoupled Weight Decay Regularization

## Start Here

1. Read `SOURCE_CARD.md` for the source boundary and project role.
2. Use `paper_pdf_pages.md` for page-anchored evidence.
3. Open the original PDF when equations, algorithms, tables, figures, or appendix details matter.

## Available Files

- `paper_pdf_pages.md`: page-by-page text extracted from the local PDF.
- `SOURCE_CARD.md`: confirmed claims, project interpretation, hypotheses, and unsupported boundary.
- `manifest.json`: provenance, checksum, extraction metadata, and derived-file list.

## alphaXiv Status

- not attempted: local PDF extraction was sufficient for this ingestion.

## Project Use

Primary source for why AdamW and Adam should be separated along the decoupled versus coupled weight-decay axis.

## Boundary

Use as evidence that L2 regularization and weight decay are not equivalent for Adam-like adaptive optimizers, and that AdamW decouples weight decay from the gradient/moment update. Do not use as direct evidence for Mahalanobis, DDU/GMM, kNN, or OOD detector behavior.

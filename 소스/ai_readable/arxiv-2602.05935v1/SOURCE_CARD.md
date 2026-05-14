# Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data

## Identity

- Source ID: `arxiv-2602.05935v1`
- arXiv ID: `2602.05935v1`
- Authors: Sudeepta Mondal, Xinyi Mary Xie, Ruxiao Duan, Alex Wong, Ganesh Sundaramoorthi
- Original PDF: `소스/Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data.pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Detector tuning protocol source for settings without a provided OOD validation set.

## Confirmed Claims

- The paper studies tuning OOD detectors without a given OOD dataset.
- It is relevant for detector-selection and tuning protocol discussions.
- It can help separate detector tuning issues from representation-geometry effects.

## Interpretation for This ICLR Project

- Useful when deciding how much OOD validation data or synthetic OOD tuning is allowed in this project's experiments.
- Useful as a protocol caution for post-hoc detector comparisons.

## Hypotheses to Test Locally

- Optimizer effects on feature geometry should be evaluated under a tuning protocol that does not accidentally tune away the geometry-sensitive failure mode.

## Unsupported Boundary

- Use as OOD-detector tuning protocol context, not as direct evidence about optimizer-induced Neural Collapse or feature-density detector mechanisms.

Boundary statement: Use as OOD-detector tuning protocol context, not as direct evidence about optimizer-induced Neural Collapse or feature-density detector mechanisms.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Treat alphaXiv material as a navigation aid only.
- Exact formulas, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

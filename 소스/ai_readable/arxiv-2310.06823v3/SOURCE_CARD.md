# NECO: Neural Collapse Based Out-of-Distribution Detection

## Identity

- Source ID: `arxiv-2310.06823v3`
- arXiv ID: `2310.06823v3`
- Authors: Mouin Ben Ammar, Nacim Belkhir, Sebastian Popescu, Antoine Manzanera, Gianni Franchi
- Original PDF: `소스/NECO_ NEURAL COLLAPSE BASED OUT-OF-DISTRIBUTION DETECTION (1).pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: External source connecting Neural Collapse geometry to post-hoc OOD scoring.

## Confirmed Claims

- The paper proposes NECO, a post-hoc OOD detector using Neural Collapse geometry and principal component spaces.
- It explicitly hypothesizes that Neural Collapse properties affecting ID data also influence OOD behavior.
- It reports experiments on small- and large-scale OOD detection tasks and claims strong generalization across architectures.

## Interpretation for This ICLR Project

- Useful as primary literature that detector-relevant geometry can be built from Neural Collapse structure.
- Helps motivate measuring NC components separately rather than treating collapse as a single scalar.

## Hypotheses to Test Locally

- If optimizers change NC geometry, NECO-like scores may change in ways that are not visible from accuracy or calibration alone.

## Unsupported Boundary

- Does not prove optimizer-induced geometry causes OOD gains or failures.
- Does not validate this repository's optimizer/weight-decay causal axis.

Boundary statement: Use as evidence that Neural Collapse geometry can be used for OOD detection, not as evidence that optimizer choice causes detector behavior.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Treat alphaXiv material as a navigation aid only.
- Exact formulas, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

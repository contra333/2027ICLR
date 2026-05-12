# Mahalanobis++: Improving OOD Detection via Feature Normalization

## Identity

- Source ID: `arxiv-2505.18032v1`
- arXiv ID: `2505.18032v1`
- Authors: Maximilian Mueller, Matthias Hein
- Original PDF: `소스/Mahalanobis++_ Improving OOD Detection via Feature Normalization (1).pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Feature-normalization control for Mahalanobis-style feature OOD detection.

## Confirmed Claims

- The paper proposes simple post-hoc L2 feature normalization before Mahalanobis scoring.
- It frames Mahalanobis inconsistency as tied to feature-norm variation and Gaussian/shared-covariance assumption violations.
- It evaluates many ImageNet-scale models and reports consistent improvement over conventional Mahalanobis scoring.

## Interpretation for This ICLR Project

- Useful for this project as a detector-side normalization/control source when studying optimizer-induced feature geometry.
- Suggests that detector performance can change because feature geometry violates or restores assumptions of the score.

## Hypotheses to Test Locally

- Optimizer-specific norm/covariance regimes in this repository may interact with Mahalanobis++ normalization strength.

## Unsupported Boundary

- Does not by itself prove optimizer choice causes Neural Collapse regimes.
- Does not replace this repository's controlled optimizer experiments.

Boundary statement: Use as evidence for post-hoc feature normalization improving Mahalanobis OOD under violated feature assumptions, not as direct optimizer-induced Neural Collapse evidence.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Treat alphaXiv material as a navigation aid only.
- Exact formulas, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

# The Marginal Value of Adaptive Gradient Methods in Machine Learning

## Identity

- Source ID: `arxiv-1705.08292v2`
- arXiv ID: `1705.08292v2`
- Authors: Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, Benjamin Recht
- Original PDF: `소스/The Marginal Value of Adaptive Gradient Methods in Machine Learning.pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: General background for adaptive optimizer implicit bias and different learned solutions.

## Confirmed Claims

- Adaptive methods such as AdaGrad, RMSProp, and Adam can find solutions that differ substantially from GD or SGD in the paper's constructed and empirical settings.
- Better training performance from adaptive methods does not necessarily imply better generalization.
- The paper reports empirical cases where SGD or SGD with momentum generalizes better than adaptive methods under comparable tuning effort.

## Interpretation for This ICLR Project

- Use this as broad support that optimizer choice can be more than a convergence-speed choice; it can affect the solution found.
- Use this as background for the project hypothesis that optimizer choice may alter representation geometry.

## PPT Use

- Supports the slide-level claim that adaptive optimizers may induce different implicit bias or solution geometry than SGD.
- Supports motivation for studying optimizer choice as an experimental axis rather than treating it as a nuisance hyperparameter.

## Hypotheses to Test Locally

- Different optimizer families may land in different feature-geometry regimes even when ID accuracy is similar.

## Unsupported Boundary

- Do not use this paper as direct evidence for Neural Collapse.
- Do not use this paper as direct evidence for Mahalanobis, DDU/GMM, kNN, Energy, or downstream OOD detector behavior.
- Do not use this paper as direct evidence for this repository's WRN/AdamW/coupling-axis results.

Boundary statement: Use as general evidence for adaptive optimizer implicit-bias and generalization differences, not as direct evidence for Neural Collapse or feature-based OOD detector behavior.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Exact formulas, algorithms, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

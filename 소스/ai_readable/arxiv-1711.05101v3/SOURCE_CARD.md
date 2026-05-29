# Decoupled Weight Decay Regularization

## Identity

- Source ID: `arxiv-1711.05101v3`
- arXiv ID: `1711.05101v3`
- Authors: Ilya Loshchilov, Frank Hutter
- Original PDF: `소스/DECOUPLED WEIGHT DECAY REGULARIZATION.pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Justifies separating Adam and AdamW by weight-decay coupling semantics.

## Confirmed Claims

- L2 regularization and weight decay are equivalent for standard SGD only under the usual learning-rate rescaling, but this equivalence does not hold for adaptive gradient algorithms such as Adam.
- AdamW decouples the weight-decay step from the gradient/moment-based optimization step.
- Decoupled weight decay can make the learning-rate choice and weight-decay-factor choice less tightly coupled in the experiments reported by the paper.

## Interpretation for This ICLR Project

- Use this as the core support for treating Adam and AdamW as different optimizer conditions, not merely variants of the same Adam baseline.
- Use this as background for the controlled coupling axis where `r=0` is AdamW-style decoupling and `r=1` is Adam-style coupling in this repository's custom optimizer.

## PPT Use

- Supports slides explaining why Adam and AdamW should be compared separately.
- Supports the motivation for a coupled-to-decoupled weight-decay axis.

## Hypotheses to Test Locally

- Coupling versus decoupling weight decay may induce different penultimate-feature geometry regimes under otherwise matched training protocols.

## Unsupported Boundary

- Do not use this paper as direct evidence that AdamW improves or worsens Mahalanobis, DDU/GMM, kNN, Energy, or other OOD detectors.
- Do not use this paper alone as proof of Neural Collapse or downstream detector behavior in this repository.

Boundary statement: Use as evidence for Adam/AdamW weight-decay coupling semantics, not as direct evidence for feature-based uncertainty detector performance.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Exact formulas, algorithms, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

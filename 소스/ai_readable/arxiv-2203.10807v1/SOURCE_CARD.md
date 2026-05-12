# ViM: Out-Of-Distribution with Virtual-logit Matching

## Identity

- Source ID: `arxiv-2203.10807v1`
- arXiv ID: `2203.10807v1`
- Authors: Haoqi Wang, Zhizhong Li, Litong Feng, Wayne Zhang
- Original PDF: `소스/ViM_ Out-Of-Distribution with Virtual-logit Matching.pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Hybrid feature-residual plus logit OOD scoring source.

## Confirmed Claims

- The paper proposes Virtual-logit Matching, combining feature-space residual information with original logits.
- It constructs a virtual OOD logit from the residual against a principal feature subspace and matches its scale to logits.
- It introduces/evaluates large-scale ImageNet OOD resources and compares CNN/vision-transformer settings.

## Interpretation for This ICLR Project

- Useful as a hybrid detector baseline because it directly combines feature geometry and logit confidence.
- Motivates reporting feature-based and logit-based detector outcomes separately in this repository.

## Hypotheses to Test Locally

- Optimizer-induced feature residual changes may affect ViM differently from pure logit or pure Mahalanobis scores.

## Unsupported Boundary

- Does not serve as Neural Collapse evidence.
- Does not prove optimizer choice is the source of feature residual changes.

Boundary statement: Use as evidence for feature-residual/logit hybrid OOD scoring, not as Neural Collapse or optimizer-causality evidence.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Treat alphaXiv material as a navigation aid only.
- Exact formulas, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

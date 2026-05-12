# OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection

## Identity

- Source ID: `arxiv-2306.09301v5`
- arXiv ID: `2306.09301v5`
- Authors: Jingyang Zhang, Jingkang Yang, Pengyun Wang, Haoqi Wang, Yueqian Lin, Haoran Zhang, Yiyou Sun, Xuefeng Du, Yixuan Li, Ziwei Liu, Yiran Chen, Hai Li
- Original PDF: `소스/OpenOOD v1.5_ Enhanced Benchmark for Out-of-Distribution Detection (2).pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Benchmark/protocol context for standardized OOD evaluation, ImageNet, and foundation-model regimes.

## Confirmed Claims

- The paper presents OpenOOD v1.5 as a standardized benchmark for OOD detection evaluation.
- It extends OpenOOD evaluation to ImageNet-scale datasets and foundation models such as CLIP and DINOv2.
- It emphasizes full-spectrum OOD detection covering semantic and covariate distribution shifts.

## Interpretation for This ICLR Project

- Useful for choosing evaluation protocol language and avoiding metric/dataset inconsistency in this project.
- Supports separating near/far semantic shift and covariate shift when reporting detector behavior.

## Hypotheses to Test Locally

- The repository's optimizer-geometry claims should be checked under OpenOOD-style standardized splits where feasible.

## Unsupported Boundary

- Does not directly support Neural Collapse or optimizer-induced geometry claims.
- Does not by itself determine which detector should be primary evidence for this project.

Boundary statement: Use as benchmark and evaluation protocol context, not as causal evidence about optimizer-induced geometry or Neural Collapse.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Treat alphaXiv material as a navigation aid only.
- Exact formulas, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

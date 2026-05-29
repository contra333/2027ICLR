# Improving Generalization Performance by Switching from Adam to SGD

## Identity

- Source ID: `arxiv-1712.07628v1`
- arXiv ID: `1712.07628v1`
- Authors: Nitish Shirish Keskar, Richard Socher
- Original PDF: `소스/Improving Generalization Performance by Switching from Adam to SGD.pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Background source for Adam-SGD comparisons in CNN/residual-family settings.

## Confirmed Claims

- The paper studies adaptive optimizers that can perform well early in training but be outperformed by SGD later in generalization.
- The paper proposes SWATS, a hybrid strategy that begins with Adam and switches to SGD under a triggering condition.
- The paper reports experiments on CIFAR-10 and CIFAR-100 with architectures including ResNet, SENet, DenseNet, and PyramidNet.

## Interpretation for This ICLR Project

- Use this as support that comparing Adam-family and SGD-family optimizers on CNN/residual architectures is not an unusual setup.
- Use this as background for explaining why fair optimizer-family comparisons should use stable update scales rather than forcing identical numerical learning rates.

## PPT Use

- Supports answering: "Is it strange to compare Adam and SGD on CNN/ResNet-style models?"
- Supports the background that Adam can train well early, while later generalization may favor SGD-family behavior.

## Hypotheses to Test Locally

- Optimizer-family update scale and training trajectory may interact with representation geometry in this repository's WRN/CIFAR experiments.

## Unsupported Boundary

- Do not use this paper as direct evidence for WRN-28-10 plus AdamW performance.
- Do not use this paper as direct evidence for Mahalanobis, DDU/GMM, kNN, Energy, or other OOD detector outcomes.
- Do not use this paper as Neural Collapse evidence.

Boundary statement: Use as evidence for Adam-SGD comparison precedent and generalization framing, not as direct evidence for this repository's OOD detector or Neural Collapse mechanism.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Exact formulas, algorithms, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

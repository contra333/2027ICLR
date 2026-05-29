# Adam: A Method for Stochastic Optimization

## Identity

- Source ID: `arxiv-1412.6980v9`
- arXiv ID: `1412.6980v9`
- Authors: Diederik P. Kingma, Jimmy Ba
- Original PDF: `소스/ADAM_ A METHOD FOR STOCHASTIC OPTIMIZATION.pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Lightweight formula/mechanics reference for Adam.

## Confirmed Claims

- Adam uses exponential moving averages of first moments and second raw moments of gradients.
- Adam applies bias correction to these moment estimates.
- Adam performs coordinate-wise adaptive updates based on the corrected moments.

## Interpretation for This ICLR Project

- Use this as a citation source for optimizer-mechanics slides when showing the Adam update equations.
- Keep this source secondary to the main optimizer-geometry and weight-decay-coupling evidence.

## PPT Use

- Supports the Adam update formula slide.
- Supports notation for `m_t`, `v_t`, bias correction, and coordinate-wise adaptive scaling.

## Hypotheses to Test Locally

- None from this source alone. This paper defines optimizer mechanics; it does not motivate the repository's feature-geometry or detector hypotheses by itself.

## Unsupported Boundary

- Do not use this paper as direct evidence for AdamW, decoupled weight decay, Neural Collapse, feature geometry, Mahalanobis, DDU/GMM, kNN, Energy, or OOD detector behavior.
- Do not treat this as a main reference-support source for the ICLR claim beyond Adam mechanics.

Boundary statement: Use as the original Adam algorithm and equation source only, not as evidence for optimizer-induced geometry or downstream detector behavior.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Exact formulas, algorithms, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

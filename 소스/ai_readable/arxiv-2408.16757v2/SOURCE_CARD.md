# Dissecting Out-of-Distribution Detection and Open-Set Recognition: A Critical Analysis of Methods and Benchmarks

## Identity

- Source ID: `arxiv-2408.16757v2`
- arXiv ID: `2408.16757v2`
- Authors: Hongjun Wang, Sagar Vaze, Kai Han
- Original PDF: `소스/Dissecting Out-of-Distribution Detection and Open-Set Recognition_ A Critical Analysis of Methods and Benchmarks.pdf`
- Page-anchored extraction: `paper_pdf_pages.md`
- Project role: Critical benchmark and method-analysis context for OOD detection and open-set recognition.

## Confirmed Claims

- The paper critically analyzes OOD detection and open-set recognition methods and benchmarks.
- It is relevant for distinguishing OOD and OSR evaluation assumptions.
- It can be used as benchmark-protocol caution rather than a source for optimizer geometry causality.

## Interpretation for This ICLR Project

- Useful as guardrail evidence when choosing evaluation protocols and avoiding benchmark-driven overclaims.
- Useful for checking whether a detector comparison is actually measuring the claimed problem setting.

## Hypotheses to Test Locally

- Some apparent optimizer-detector effects may depend on the OOD/OSR benchmark framing and should be stress-tested across protocols.

## Unsupported Boundary

- Use as OOD/OSR benchmark and method-analysis context, not as direct Neural Collapse, optimizer-causality, or feature-geometry evidence.

Boundary statement: Use as OOD/OSR benchmark and method-analysis context, not as direct Neural Collapse, optimizer-causality, or feature-geometry evidence.

## Evidence Policy

- Treat `paper_pdf_pages.md` and the original PDF as primary evidence for exact claims.
- Treat alphaXiv material as a navigation aid only.
- Exact formulas, tables, appendix settings, and reproduction-critical details require checking the original PDF page.

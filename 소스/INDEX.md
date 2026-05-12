# 소스/INDEX.md

Last updated: 2026-05-12 KST

## Purpose

This index tells AI agents what sources exist in `소스/`, what each source can support, and where to look before reading large files.

Read this file before opening full PDFs or large extracted text.

## Current inventory

| File | Role | Evidence use |
|---|---|---|
| `2027_ICLR_실험레포설계.md` | Main ICLR project design and experiment roadmap | Project planning, priorities, risks, experiment staging |
| `neurips2026_paper_context.md` | AI-readable context for the user's NeurIPS 2026 paper | Primary navigation file for the prior NeurIPS claim structure |
| `0_neurips_2026.pdf` | NeurIPS 2026 submitted paper PDF copy | Original visual source for the user's prior paper |
| `2026NeurIPS주제를 2027ICLR로 발전시킬 교수님디렉션.md` | Professor/project direction note | Project direction and interpretation, not a metric source |
| `GPT_PROJECT_SOURCE_GUIDE.md` | Guide for using arXiv 2602.16642v3 | Start here for the Neural Collapse optimizer paper |
| `context_manifest.json` | Manifest for the arXiv 2602.16642v3 source package | Provenance, extraction metadata, evidence tier policy |
| `SRC-arxiv-2602.16642-v3.md` | Source card for `Optimizer choice matters for the emergence of Neural Collapse` | Source boundary and supported/unsupported claims |
| `paper_pdf_pages.md` | Page-anchored PDF text for arXiv 2602.16642v3 | Primary page context for claims when TeX source is not available locally |
| `paper.pdf` | PDF copy for arXiv 2602.16642v3 | Original layout/equation/table source for that paper |
| `SOURCE_MANIFEST_TEMPLATE.json` | Template for future source manifests | Use when ingesting new sources into this folder |

## Important absence

`GPT_PROJECT_SOURCE_GUIDE.md` mentions `paper_source_tex_combined.md`, but that file is not currently present in this local `소스/` folder.

Do not claim to have checked the TeX source from this repository unless that file is added or retrieved from the recorded version directory.

## Source boundaries

### NeurIPS 2026 paper

Supported by the user's NeurIPS paper context:

- SAM can improve or preserve accuracy/calibration/logit-level reliability while hurting feature-based uncertainty detectors.
- The proposed mechanism is penultimate feature geometry change: within-class dispersion, covariance inflation, and detector-relevant class structure changes.
- Mahalanobis and DDU rely on feature class means/covariances rather than just output confidence.

Use `neurips2026_paper_context.md` first, then the PDF when exact layout or wording matters.

### Optimizer choice / Neural Collapse paper

Supported by this source package:

- optimizer choice affects Neural Collapse / representation geometry,
- coupled versus decoupled weight decay is a meaningful intervention axis,
- NC metrics can split, so geometry should not be reduced to one scalar.

Not directly supported by this paper:

- Energy score failure or success,
- Mahalanobis degradation,
- kNN, DDU, or feature-density detector behavior,
- causal proof that NC changes cause downstream uncertainty divergence.

## Recommended reading order by task

- Project orientation: `AI_CONTEXT.md` -> this index -> `2027_ICLR_실험레포설계.md`.
- NeurIPS prior-work claim: `neurips2026_paper_context.md` -> `0_neurips_2026.pdf` if exact wording is needed.
- Neural Collapse optimizer source: `GPT_PROJECT_SOURCE_GUIDE.md` -> `SRC-arxiv-2602.16642-v3.md` -> `context_manifest.json` -> `paper_pdf_pages.md` targeted pages.
- Source ingestion: `소스/AGENTS.md` -> relevant source file -> manifest/source card update -> this index.

## Claim handling template

When using a source, write claims in this style:

- Claim:
- Status: `confirmed`, `interpretation`, `hypothesis`, or `unsupported`
- Evidence path:
- Page/section:
- Notes:

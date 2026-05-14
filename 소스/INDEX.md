# 소스/INDEX.md

Last updated: 2026-05-13 KST

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
| `paper.pdf` + `ai_readable/arxiv-2602.16642v3/AI_README.md` | Original PDF and AI-readable source package for arXiv 2602.16642v3 | Optimizer choice / Neural Collapse / representation geometry evidence; not downstream OOD detector evidence |
| `A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts.pdf` + `ai_readable/arxiv-2511.11934v2/AI_README.md` | Local PDF and AI-readable source package | Representation/training paradigm shifts in OOD detection; nearby context, not optimizer/NC causality evidence |
| `Dissecting Out-of-Distribution Detection and Open-Set Recognition_ A Critical Analysis of Methods and Benchmarks.pdf` + `ai_readable/arxiv-2408.16757v2/AI_README.md` | Local PDF and AI-readable source package | OOD/OSR method and benchmark analysis; protocol context |
| `Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era_ A Survey.pdf` + `ai_readable/arxiv-2407.21794v2/AI_README.md` | Local PDF and AI-readable source package | Generalized OOD and VLM-era survey context |
| `Mahalanobis++_ Improving OOD Detection via Feature Normalization (1).pdf` + `ai_readable/arxiv-2505.18032v1/AI_README.md` | Mahalanobis++ local PDF and AI-readable source package | Feature normalization / Mahalanobis OOD evidence; not optimizer-causality evidence |
| `NECO_ NEURAL COLLAPSE BASED OUT-OF-DISTRIBUTION DETECTION (1).pdf` + `ai_readable/arxiv-2310.06823v3/AI_README.md` | NECO local PDF and AI-readable source package | Neural Collapse geometry used for OOD scoring; not evidence that optimizer choice causes detector behavior |
| `One Model, Many Behaviors_ Training-Induced Effects on Out-of-Distribution Detection.pdf` + `ai_readable/arxiv-2601.10836v1/AI_README.md` | Local PDF and AI-readable source package | Training-induced OOD behavior context; adjacent motivation for controlled training-regime comparisons |
| `OpenOOD v1.5_ Enhanced Benchmark for Out-of-Distribution Detection (2).pdf` + `ai_readable/arxiv-2306.09301v5/AI_README.md` | OpenOOD v1.5 local PDF and AI-readable source package | Benchmark/protocol context for standardized OOD evaluation |
| `Out-of-Distribution Detection_ A Task-Oriented Survey of Recent Advances.pdf` + `ai_readable/arxiv-2409.11884v4/AI_README.md` | Local PDF and AI-readable source package | Task-oriented OOD survey and detector taxonomy context |
| `Recent Advances in Out-of-Distribution Detection with CLIP-Like Models_ A Survey.pdf` + `ai_readable/arxiv-2505.02448v1/AI_README.md` | Local PDF and AI-readable source package | CLIP-like model OOD survey context for pretrained/foundation-model regimes |
| `Tuning Out-of-Distribution (OOD) Detectors Without Given OOD Data.pdf` + `ai_readable/arxiv-2602.05935v1/AI_README.md` | Local PDF and AI-readable source package | OOD detector tuning protocol context when no OOD validation data is given |
| `ViM_ Out-Of-Distribution with Virtual-logit Matching.pdf` + `ai_readable/arxiv-2203.10807v1/AI_README.md` | ViM local PDF and AI-readable source package | Feature-residual plus logit hybrid OOD scoring evidence; not Neural Collapse evidence |
| `SOURCE_MANIFEST_TEMPLATE.json` | Template for future source manifests | Use when ingesting new sources into this folder |

## Important absence

The local arXiv 2602.16642v3 package does not currently include TeX source or a combined TeX-derived text file such as `paper_source_tex_combined.md`.

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

### Mahalanobis++ / feature normalization

Supported by this source package:

- post-hoc L2 feature normalization can improve Mahalanobis-style OOD detection,
- Mahalanobis performance can depend on feature-norm variation and Gaussian/shared-covariance assumption fit,
- feature normalization is a detector-side control to consider when interpreting geometry-sensitive scores.

Not directly supported by this paper:

- optimizer choice causes Neural Collapse regimes,
- optimizer-induced geometry is the reason Mahalanobis succeeds or fails in this repository.

### NECO / Neural Collapse OOD

Supported by this source package:

- Neural Collapse geometry can be used as the basis for a post-hoc OOD detector,
- principal-component geometry and collapse-like structure can be detector-relevant.

Not directly supported by this paper:

- optimizer choice causes detector behavior,
- this repository's optimizer/weight-decay axis is causally validated.

### OpenOOD v1.5

Supported by this source package:

- standardized OOD benchmark and evaluation protocol context,
- ImageNet-scale and foundation-model OOD evaluation framing,
- full-spectrum OOD detection across semantic and covariate shifts.

Not directly supported by this paper:

- Neural Collapse or optimizer-induced geometry claims,
- which detector should be primary evidence for this project.

### ViM

Supported by this source package:

- hybrid OOD scoring using both feature residuals and logits,
- principal feature subspace residuals can provide class-agnostic OOD signal.

Not directly supported by this paper:

- Neural Collapse evidence,
- optimizer choice as the cause of feature residual changes.

### Representation / training-shift OOD sources

Covered source packages:

- `ai_readable/arxiv-2511.11934v2/AI_README.md`
- `ai_readable/arxiv-2601.10836v1/AI_README.md`

Supported by these source packages:

- OOD detector behavior can be studied under representation shifts, training paradigm shifts, or training-induced behavioral changes.
- These papers are useful as nearby motivation for controlled comparisons across training regimes, representations, and detector families.

Not directly supported by these papers:

- this repository's specific optimizer -> Neural Collapse -> detector mechanism,
- causal proof that Mahalanobis, DDU/GMM, or kNN behavior changes because of the NC metrics used in this project.

### OOD / OSR benchmark-analysis and task surveys

Covered source packages:

- `ai_readable/arxiv-2408.16757v2/AI_README.md`
- `ai_readable/arxiv-2409.11884v4/AI_README.md`

Supported by these source packages:

- OOD/OSR method, benchmark, and task-framing context,
- detector-family taxonomy and protocol cautions for reporting.

Not directly supported by these papers:

- optimizer-induced representation geometry,
- Neural Collapse as the mechanism behind feature-based uncertainty behavior.

### VLM and CLIP-like OOD surveys

Covered source packages:

- `ai_readable/arxiv-2407.21794v2/AI_README.md`
- `ai_readable/arxiv-2505.02448v1/AI_README.md`

Supported by these source packages:

- generalized OOD and CLIP/VLM-era OOD detection context,
- framing for pretrained or foundation-model regimes.

Not directly supported by these papers:

- controlled CNN optimizer causality,
- direct evidence for Neural Collapse changes causing downstream detector behavior.

### OOD detector tuning without given OOD data

Supported by this source package:

- detector tuning protocol context when no OOD validation data is provided,
- cautions for separating detector tuning effects from representation-geometry effects.

Not directly supported by this paper:

- optimizer-induced Neural Collapse,
- causal evidence that a feature-density detector succeeds or fails because of optimizer choice.

## Recommended reading order by task

- Project orientation: `AI_CONTEXT.md` -> this index -> `2027_ICLR_실험레포설계.md`.
- NeurIPS prior-work claim: `neurips2026_paper_context.md` -> `0_neurips_2026.pdf` if exact wording is needed.
- Neural Collapse optimizer source: `ai_readable/arxiv-2602.16642v3/AI_README.md` -> `ai_readable/arxiv-2602.16642v3/GPT_PROJECT_SOURCE_GUIDE.md` -> `ai_readable/arxiv-2602.16642v3/SRC-arxiv-2602.16642-v3.md` -> `ai_readable/arxiv-2602.16642v3/context_manifest.json` -> targeted `ai_readable/arxiv-2602.16642v3/paper_pdf_pages.md` pages.
- OOD/NC detector, survey, and protocol source packages: start with each `ai_readable/arxiv-*/AI_README.md` -> `SOURCE_CARD.md` -> `paper_pdf_pages.md` targeted pages -> original PDF for exact equations/tables/figures.
- Source ingestion: `소스/AGENTS.md` -> relevant source file -> manifest/source card update -> this index.

## Claim handling template

When using a source, write claims in this style:

- Claim:
- Status: `confirmed`, `interpretation`, `hypothesis`, or `unsupported`
- Evidence path:
- Page/section:
- Notes:

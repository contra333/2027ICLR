---
id: SRC-arxiv-2602.16642-v3
type: source
source_type: paper
status: ingested
title: "Optimizer choice matters for the emergence of Neural Collapse"
source_path: "20_원본자료/논문/arxiv-2602.16642/versions/v3"
origin_path: "20_원본자료/논문/arxiv-2602.16642/versions/v3"
arxiv_base_id: "2602.16642"
arxiv_version: "v3"
manifest_path: "20_원본자료/논문/arxiv-2602.16642/versions/v3/extraction_manifest.json"
summary: "Optimizer choice and weight-decay coupling change the emergence of neural collapse; this source supports the optimizer -> geometry axis, not the downstream uncertainty divergence axis."
key_facts:
  - "The paper challenges optimizer-independent universality of Neural Collapse."
  - "The arXiv TeX source states that adaptive optimizers with decoupled weight decay, including AdamW, have much larger NC metrics and no sign of NC in realistic settings."
  - "The paper includes an AdamW-to-Adam interpolation where increasing coupled weight decay improves NC0, NC2, and NC3 while validation accuracy remains largely unaffected."
claim_candidates:
  - "Optimizer design can create distinct representation geometry regimes."
  - "Coupled versus decoupled weight decay is a meaningful geometry intervention axis."
limitations:
  - "This paper does not directly evaluate downstream uncertainty families such as Energy, Mahalanobis, kNN, or DDU."
  - "Use it as primary evidence for optimizer -> NC/geometry, not for feature-based OOD degradation."
notes: "AlphaXiv fulltext is secondary context; supported claims should cite arXiv TeX or PDF page context."
topics:
  - optimizer-induced-geometry
  - neural-collapse
  - weight-decay-coupling
claims:
  - CLM-optimizer-induced-uncertainty-divergence
  - CLM-geometry-not-single-nc-score
concepts:
  - optimizer-induced-geometry-regime
created: 2026-04-30
updated: 2026-04-30
---

# Source: Optimizer choice matters for the emergence of Neural Collapse

## What this source is

- arXiv paper ingestion folder for `2602.16642v3`.
- AlphaXiv files, if present, are secondary evidence only.
- Exact equations, tables, hyperparameters, datasets, metrics, and results require primary evidence.

## Evidence Policy

| status | meaning |
|---|---|
| candidate | proposed from secondary or incomplete evidence |
| supported | backed by primary evidence |
| rejected | contradicted by primary evidence |
| unknown | evidence not found |

Required fields for extracted claims or experiment settings:

- status
- value
- evidence_type
- evidence_path
- page
- section
- quote_or_context
- confidence
- notes

## Available Artifacts

- Manifest: `20_원본자료/논문/arxiv-2602.16642/versions/v3/extraction_manifest.json`
- Version folder: `20_원본자료/논문/arxiv-2602.16642/versions/v3`

## Extracted Evidence For This Project

| Claim use | Status | Evidence type | Evidence path | Section | Context |
|---|---|---|---|---|---|
| optimizer choice affects NC/geometry | supported | arXiv TeX source | `20_원본자료/논문/arxiv-2602.16642/versions/v3/derived/source_tex/sections/main_result.tex` | Experimental Setup; Weight Decay Coupling Matters | The TeX source reports Adam/AdamW/SGD/SGDW comparisons across 3,888+ runs and states that adaptive optimizers with decoupled WD have larger NC metrics. |
| AdamW-to-Adam interpolation is a useful controlled transition | supported | arXiv TeX source | `20_원본자료/논문/arxiv-2602.16642/versions/v3/derived/source_tex/sections/main_result.tex` | Interpolating AdamW and Adam | Increasing the coupled WD component smoothly improves NC0, NC2, NC3 while validation accuracy remains largely unaffected. |
| downstream uncertainty divergence | candidate | scope boundary | this source does not test it | N/A | This is the gap our NeurIPS project can occupy. |

## Limitations / uncertainty

- PDF text extraction can distort equations, tables, figure captions, and two-column layout.
- If a detail is missing, record `UNKNOWN / NOT FOUND IN AVAILABLE SOURCES`.

## Do not overclaim

- Do not mark a claim as `supported` unless it has primary evidence: arXiv TeX, PDF page context, official code/config, experiment log, plot script, or result file.

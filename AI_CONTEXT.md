# AI_CONTEXT.md

Last updated: 2026-05-12 KST

## One-sentence project context

This project develops the user's NeurIPS 2026 paper into an ICLR 2027 submission about optimizer-induced feature geometry regimes and their consequences for feature-based uncertainty/OOD detection.

## Current thesis

The intended ICLR story is not simply "Neural Collapse is good" or "SAM is bad." The stronger thesis is:

Optimizer improvements do not have a single reliability meaning. Optimizers induce distinct penultimate geometry regimes, and feature-based uncertainty succeeds or fails depending on which parts of Neural Collapse-like geometry are preserved, distorted, or masked.

The target title direction is:

Neural Collapse Is Not One Geometry: Optimizer-Dependent Collapse Regimes and Their Consequences for OOD Detection

## What is already in this repo

- `소스/2027_ICLR_실험레포설계.md`: main planning document for the ICLR project and experiment roadmap.
- `소스/neurips2026_paper_context.md`: AI-readable context for the user's NeurIPS 2026 submission.
- `소스/0_neurips_2026.pdf`: original NeurIPS 2026 paper PDF copy.
- `소스/GPT_PROJECT_SOURCE_GUIDE.md`, `소스/SRC-arxiv-2602.16642-v3.md`, `소스/context_manifest.json`, `소스/paper_pdf_pages.md`, `소스/paper.pdf`: source package for `Optimizer choice matters for the emergence of Neural Collapse`.
- `AGENTS.md`: root AI operating rules.
- `소스/INDEX.md`: source inventory and evidence boundaries.
- `ops/`: multi-server Git, server run, and result-sync operating instructions.
- `reports/METRIC_DEFINITIONS.md`: metric contract for server evaluators, Codex CLI, and GPT-based analysis; it uses revised NC names such as `nc0_width_norm`, `nc3_self_duality`, `nc4_agreement`, and `inter_dist_l2` to avoid legacy-name ambiguity.

## Evidence boundaries to preserve

- NeurIPS 2026 claim: vanilla SAM can preserve or improve accuracy/calibration/logit-level reliability while degrading Mahalanobis/DDU through penultimate feature geometry changes.
- Neural Collapse optimizer paper claim: optimizer choice and coupled/decoupled weight decay affect NC / representation geometry.
- ICLR hypothesis: optimizer-induced NC/feature geometry regimes explain downstream feature-based uncertainty detector behavior across CNNs, ViTs, and pretrained regimes.

Do not merge these into one unsupported claim. The downstream detector link needs this project's own experiments or another primary source.

## Intended experiment roadmap

1. Standard CNN main evidence:
   - CIFAR-10/100, standard ResNet-18 and WRN-28-10.
   - No DDU-specific spectral normalization or modified activations for the main claim.
   - Compare SGD, SGDW, Adam, AdamW, SAM, ASAM, GSAM.
   - Track accuracy, calibration, logit-based OOD, feature-based OOD, and geometry metrics.
2. AdamW-to-Adam / coupled-decoupled WD axis:
   - Sweep coupled versus decoupled weight decay while monitoring accuracy, NC metrics, Mahalanobis, DDU/GMM, and kNN.
3. DDU-style architecture diagnostics:
   - Treat SN/mod settings from the old NeurIPS/DDU code as appendix or diagnostic, not the main ICLR evidence.
4. ViT extension:
   - Start with CIFAR-scale ViT-Tiny/DeiT-Tiny from scratch.
   - Compare CLS, mean-pooled, and pre-logit features.
   - Treat LayerNorm/bias weight-decay policy as an explicit experimental axis.
5. Pretrained regime:
   - Use frozen/linear-probe/full-finetune pretrained ViT or CLIP/DINO features as a regime analysis, not the main proof of optimizer-induced geometry.

## Current repo status

- This workspace is currently a research operations and source context repo.
- Git has been initialized locally on branch `main`, with `origin` set to `https://github.com/contra333/2027ICLR.git`.
- Real GPU training code may live on a server and later be mirrored or imported into `code/`.
- The intended execution pattern is local planning/analysis plus GPU execution on shared servers such as `101`, `175`, and `138`.
- All servers should use the same private Git remote when available. Git is for code/config/docs/manifests; large raw outputs and checkpoints are copied outside Git tracking.
- Server results should be copied into `results/raw/<run_id>/` with manifests under `results/manifests/`.
- Reports and professor-facing summaries should go under `reports/`.
- One-off work requests should be captured under `tasks/` when they need durable tracking.

## Known risks

- DDU alone is not enough to represent all feature-based OOD behavior; include Mahalanobis, kNN, tied/diagonal/shrinkage GMM, and feature normalization controls.
- Full covariance DDU/GMM can be unstable for CIFAR-100, ViT, or high-dimensional pretrained features.
- Results from SN/mod DDU architectures can be confounded with optimizer effects.
- ViT results may be noisier or regime-dependent; CNN controlled evidence should carry the main claim.
- Pretraining may mask optimizer-induced geometry shifts, which should be framed as a regime finding rather than a failure.

## Next actions

- Add or mirror the first minimal experiment code into `code/`.
- Define baseline configs under `configs/` before running server jobs.
- Use `reports/METRIC_DEFINITIONS.md` before implementing or interpreting evaluator outputs.
- New evaluator code should emit revised metric names instead of legacy `nc0`, `nc3`, `nc4`, or `inter_dist` names.
- For each server run, copy back raw outputs and write a `results/manifests/*.json` file.
- Clone the GitHub repo on `101`, `175`, and `138` before server-side experiment implementation.
- Use `ops/MULTI_SERVER_GIT_WORKFLOW.md`, `ops/SERVER_RUN_TEMPLATE.md`, and `ops/RESULT_SYNC_GUIDE.md` when preparing the first server run.
- Keep `AI_CONTEXT.md` updated after major milestones so new Codex sessions can restart cheaply.

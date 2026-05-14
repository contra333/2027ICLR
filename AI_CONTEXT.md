# AI_CONTEXT.md

Last updated: 2026-05-13 KST

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
- `소스/paper.pdf` and `소스/ai_readable/arxiv-2602.16642v3/`: original PDF and AI-readable source package for `Optimizer choice matters for the emergence of Neural Collapse`.
- `AGENTS.md`: root AI operating rules.
- `소스/INDEX.md`: source inventory and evidence boundaries.
- `ops/`: multi-server Git, server run, and result-sync operating instructions.
- `reports/METRIC_DEFINITIONS.md`: metric contract for server evaluators, Codex CLI, and GPT-based analysis; it uses revised NC names such as `nc0_width_norm`, `nc3_self_duality`, `nc4_agreement`, and `inter_dist_l2` to avoid legacy-name ambiguity.
- `reports/DAILY_REPORT_WORKFLOW.md` and paired daily reports such as `reports/MMDD_<topic>.md` / `.html`: chronological daily work records. Future AI sessions should use these reports, in addition to `AI_CONTEXT.md`, to reconstruct what happened on each date, what files were read, what decisions were made, and what remains unresolved.
- `ops/SERVER_CLONE_TO_FIRST_RUN.md`: server clone-to-smoke-run workflow for `101`, `175`, and `138`.
- `code/IMPLEMENTATION_CONTRACT.md`: server code contract for standard training, optimizer semantics, and shared-cache evaluation.
- `code/models/ARCHITECTURE_CONTRACT.md` and `reports/ARCHITECTURE_REFERENCES.md`: standard CIFAR ResNet/WRN definitions and implementation references.

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

- This workspace is now a research operations, source context, and minimal experiment-code repo.
- Git has local branch `main` for stable shared state and `exp/m1-smoke-pipeline` for the M1 smoke implementation work.
- `origin` is set to `https://github.com/contra333/2027ICLR.git`.
- Minimal standard CIFAR training/evaluation code is present under `code/`.
- The intended execution pattern is local planning/analysis plus GPU execution on shared servers such as `101`, `175`, and `138`.
- All servers should use the same private Git remote when available. Git is for code/config/docs/manifests; large raw outputs and checkpoints are copied outside Git tracking.
- Server results should be copied into `results/raw/<run_id>/` with manifests under `results/manifests/`.
- Reports and professor-facing summaries should go under `reports/`.
- One-off work requests should be captured under `tasks/` when they need durable tracking.

## Recent milestone status

### M1A smoke pipeline

Status: completed for the 2 epoch CIFAR-10 `standard_cifar_resnet18` SGD smoke run.

Confirmed run:

- `20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0`

Registered manifest:

- `results/manifests/20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0.json`

The run completed training, shared-cache extraction, post-hoc evaluation, and `code/tests/smoke_checks.py --check run-dir`. Checkpoint and cache `.pt` artifacts remain under `/home/ghjin/iclr2027_runs/<run_id>/`; only metrics/log/config snapshots were copied into `results/raw/<run_id>/`, which is ignored by Git.

### M1B optimizer endpoint smoke validation

Status: completed for 2 epoch smoke runs of:

- `adam`
- `adamw`
- `adam_coupled_decoupled` with `coupled_ratio: 0.0`
- `adam_coupled_decoupled` with `coupled_ratio: 1.0`

Implemented files:

- `code/optimizers/adam_coupled_decoupled.py`
- `code/optimizers/factory.py`
- `code/optimizers/__init__.py`
- `code/tests/smoke_checks.py`

Added smoke configs:

- `configs/smoke/cifar10_standard-cifar-resnet18_adam_2ep_seed0.yaml`
- `configs/smoke/cifar10_standard-cifar-resnet18_adamw_2ep_seed0.yaml`
- `configs/smoke/cifar10_standard-cifar-resnet18_adam-coupled-decoupled_r0_2ep_seed0.yaml`
- `configs/smoke/cifar10_standard-cifar-resnet18_adam-coupled-decoupled_r1_2ep_seed0.yaml`

Endpoint check:

- `code/tests/smoke_checks.py --check optimizer-endpoints` verifies that `coupled_ratio=0.0` matches PyTorch AdamW one-step behavior and `coupled_ratio=1.0` matches PyTorch Adam one-step behavior on a toy parameter-group setup.

Registered M1B manifests:

- local-curie validation: `20260512_2334` through `20260512_2337`
- SSH host `101` validation: `20260512_2356` through `20260512_2359`

These are smoke validations only. Their 2 epoch metrics confirm the pipeline and optimizer endpoint plumbing, not paper-level evidence.

Endpoint interpretation note:

- `reports/OPTIMIZER_ENDPOINT_SEMANTICS_2026-05-13.md` records the policy for Adam/AdamW versus `adam_coupled_decoupled` endpoints.
- Use the custom `adam_coupled_decoupled` implementation as the controlled interpolation axis.
- Treat `coupled_ratio=0.0` as AdamW-style and `coupled_ratio=1.0` as Adam-style, but do not claim full training trajectories are bitwise-identical to PyTorch `AdamW` or `Adam`.

## Known risks

- DDU alone is not enough to represent all feature-based OOD behavior; include Mahalanobis, kNN, tied/diagonal/shrinkage GMM, and feature normalization controls.
- Full covariance DDU/GMM can be unstable for CIFAR-100, ViT, or high-dimensional pretrained features.
- Results from SN/mod DDU architectures can be confounded with optimizer effects.
- ViT results may be noisier or regime-dependent; CNN controlled evidence should carry the main claim.
- Pretraining may mask optimizer-induced geometry shifts, which should be framed as a regime finding rather than a failure.
- Optimizer endpoint interpretation can be confused: PyTorch `AdamW`/`Adam` are baseline anchors, while `adam_coupled_decoupled` endpoints are controlled interpolation endpoints. See `reports/OPTIMIZER_ENDPOINT_SEMANTICS_2026-05-13.md`.

## Next actions

- M1A/M1B smoke code, configs, task notes, reports, and manifests have been pushed to `main`.
- Treat the current M1 smoke metrics as pipeline validation only; do not interpret them as ICLR evidence.
- Preserve the optimizer endpoint semantics policy before designing the first 200 epoch sweep.
- Before any 200 epoch sweep, prepare matched-protocol configs and decide the first optimizer set and server allocation.
- Keep using `reports/METRIC_DEFINITIONS.md` before interpreting evaluator outputs.
- Keep emitting revised metric names instead of legacy `nc0`, `nc3`, `nc4`, or `inter_dist` names.
- For each server run, copy back metrics/log/config snapshots and write a `results/manifests/*.json` file.
- Use `ops/MULTI_SERVER_GIT_WORKFLOW.md`, `ops/SERVER_RUN_TEMPLATE.md`, `ops/RESULT_SYNC_GUIDE.md`, and `ops/RUN_MANIFEST_RULES.md` when preparing or syncing server runs.
- Keep `AI_CONTEXT.md` updated after major milestones so new Codex sessions can restart cheaply.

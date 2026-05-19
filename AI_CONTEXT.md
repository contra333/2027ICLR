# AI_CONTEXT.md

Last updated: 2026-05-19 KST

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
- `0519코드수정지침.md`, `0519지침에따른수정계획.md`, and `학습계획.md` in the parent workspace: May 19 implementation guidance, code-change plan, and WRN-28-10 350 epoch training plan. These files are outside the repo root but should be consulted before changing the May 19 training/evaluation setup.
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

### May 19 WRN350 training/evaluation preparation

Status: code prepared locally for the first paper-scale WRN anchor, but no 350 epoch run has been launched from this repo yet.

Implemented or updated:

- `code/models/wide_resnet.py` and `code/models/factory.py`: standard WRN-28-10 registry entries, including `standard_wrn_28_10_dropout03`.
- `configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml`: first main anchor config for CIFAR-10, WRN-28-10 dropout 0.3, SGD, 350 epochs, MultiStepLR milestones `[150, 250]`, checkpoint every 50 epochs, final checkpoint, and best-val checkpoint.
- `code/data/registry.py`, `code/data/factory.py`, `code/data/transforms.py`, and `code/data/splits.py`: config-driven dataset registry, CIFAR/SVHN/MNIST/TinyImageNet loader support, ID-normalized OOD transforms, stratified split metadata, and class-count metadata.
- `code/eval/logit_detectors.py`, `code/eval/feature_detectors.py`, `code/eval/geometry.py`, `code/eval/aggregate_metrics.py`, and `code/eval_posthoc.py`: added or wired `maxlogit`, `neg_entropy`, `knn_l2`, registry-based detector selection, revised geometry metrics, detector metadata, and stricter unknown-name failure.
- `code/train.py` and `code/extract_cache.py`: checkpoint policy and checkpoint-tagged cache extraction for post-hoc evaluation.
- `reports/METRIC_DEFINITIONS.md` and `code/IMPLEMENTATION_CONTRACT.md`: metric/evaluator contract updates matching the May 19 code.

Validation already run on the May 19 code before pushing:

- `python code/tests/smoke_checks.py --config configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml --check model`
- `python code/tests/smoke_checks.py --config configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml --check registry`
- `python code/tests/smoke_checks.py --config configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml --check optimizer-endpoints`
- `python code/tests/smoke_checks.py --config configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml --check unit-metrics`

Current execution recommendation:

1. Push or pull the May 19 code on a GPU server with free capacity.
2. Prepare datasets under `${HOME}/datasets` before full config execution. CIFAR-10, CIFAR-100, SVHN, and MNIST use torchvision download paths, but TinyImageNet is an ImageFolder-style dataset and must exist at `${HOME}/datasets/tiny-imagenet-200/val` unless the config is changed.
3. Run only the SGD seed0 WRN350 anchor first. Complete train -> cache extraction -> post-hoc eval -> `smoke_checks.py --check run-dir` before expanding to Adam/AdamW/coupled-decoupled sweeps.
4. Do not tune detector or optimizer hyperparameters on OOD test metrics. Use fixed config or ID validation only.

May 19 server observation:

- The active server seen from this workspace was `curie`, with four RTX A5000 GPUs fully occupied by unrelated long-running jobs. Do not launch the WRN350 anchor there until capacity is free; use another server if available.
- `/home/ghjin/datasets` on that server contained CIFAR-10 and SVHN `test_32x32.mat`, but not CIFAR-100, MNIST, or TinyImageNet. Full WRN config data loading failed with `download=false` because CIFAR-100 was missing.

## Known risks

- DDU alone is not enough to represent all feature-based OOD behavior; include Mahalanobis, kNN, tied/diagonal/shrinkage GMM, and feature normalization controls.
- Full covariance DDU/GMM can be unstable for CIFAR-100, ViT, or high-dimensional pretrained features.
- Results from SN/mod DDU architectures can be confounded with optimizer effects.
- ViT results may be noisier or regime-dependent; CNN controlled evidence should carry the main claim.
- Pretraining may mask optimizer-induced geometry shifts, which should be framed as a regime finding rather than a failure.
- Optimizer endpoint interpretation can be confused: PyTorch `AdamW`/`Adam` are baseline anchors, while `adam_coupled_decoupled` endpoints are controlled interpolation endpoints. See `reports/OPTIMIZER_ENDPOINT_SEMANTICS_2026-05-13.md`.
- The current WRN350 SGD anchor config has `optimizer.nesterov: false`. The May 19 training plan notes that `true` is closer to original WRN practice, while `false` preserves continuity with earlier ResNet smoke settings. Decide before launching the definitive anchor.
- The May 19 full config includes TinyImageNet as an OOD dataset, but TinyImageNet is not auto-downloaded by the current code. Missing ImageFolder data will block `build_data_bundle`.

## Next actions

- M1A/M1B smoke code, configs, task notes, reports, and manifests have been pushed to `main`.
- Treat the current M1 smoke metrics as pipeline validation only; do not interpret them as ICLR evidence.
- Push the May 19 WRN350 code/context update to `main`, then pull it from an available GPU server.
- Prepare the target server datasets, especially CIFAR-100, MNIST, and TinyImageNet, before launching the full WRN350 config.
- Run the WRN-28-10 dropout03 CIFAR-10 SGD seed0 350 epoch anchor first, then evaluate final and selected checkpoint tags through shared-cache extraction and post-hoc metrics.
- Preserve the optimizer endpoint semantics policy before expanding from the SGD anchor to Adam/AdamW/coupled-decoupled sweeps.
- Keep using `reports/METRIC_DEFINITIONS.md` before interpreting evaluator outputs.
- Keep emitting revised metric names instead of legacy `nc0`, `nc3`, `nc4`, or `inter_dist` names.
- For each server run, copy back metrics/log/config snapshots and write a `results/manifests/*.json` file.
- Use `ops/MULTI_SERVER_GIT_WORKFLOW.md`, `ops/SERVER_RUN_TEMPLATE.md`, `ops/RESULT_SYNC_GUIDE.md`, and `ops/RUN_MANIFEST_RULES.md` when preparing or syncing server runs.
- Keep `AI_CONTEXT.md` updated after major milestones so new Codex sessions can restart cheaply.

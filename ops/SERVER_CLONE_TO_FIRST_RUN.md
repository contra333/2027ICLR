# Server Clone to First Run Workflow

Last updated: 2026-05-12 KST

## Purpose

This file tells a server-side Codex or human operator what to do immediately after cloning `2027ICLR` on a GPU server.

The first goal is not to produce paper results. The first goal is to verify the full training -> cache extraction -> post-hoc evaluation -> manifest -> local sync pipeline.

## Required clone

Clone only the `2027ICLR` repository first:

```bash
git clone https://github.com/contra333/2027ICLR.git 2027ICLR
cd 2027ICLR
git checkout main
git pull --ff-only
```

`DDU_fork` is not required for Milestone 1. Treat it as a reference for DDU-style spectral normalization, mod settings, and legacy DDU/GMM diagnostics only when those diagnostics are implemented later.

## Read before coding

Read these files in this order:

1. `AGENTS.md`
2. `AI_CONTEXT.md`
3. `ops/SERVER_CLONE_TO_FIRST_RUN.md`
4. `code/IMPLEMENTATION_CONTRACT.md`
5. `code/models/ARCHITECTURE_CONTRACT.md`
6. `reports/METRIC_DEFINITIONS.md`

For config work, also read `configs/AGENTS.md`. For result import or interpretation, also read `results/AGENTS.md`.

## Server output root

Prefer an output root outside the Git repo:

```bash
mkdir -p "$HOME/iclr2027_runs"
```

If output must be under the repo, use ignored paths only:

```bash
mkdir -p results/raw results/checkpoints results/tmp
```

Do not commit checkpoints, feature dumps, raw run folders, large logs, datasets, or local environments.

## Run ID rule

Use this run ID format:

```text
YYYYMMDD_HHMM_<server>_<dataset>_<model>_<optimizer>_seed<seed>
```

Example:

```text
20260512_2130_101_cifar10_standard-cifar-resnet18_sgd_seed0
```

Every run directory must include:

- `git_commit.txt`
- `config.sha256`
- `config_snapshot.yaml`
- `command.txt`
- `run.log`
- training metrics or evaluator metrics
- checkpoint paths or an explicit checkpoint policy

## Milestone 1: pipeline smoke run

Implement only enough code to validate the pipeline:

- dataset: CIFAR-10
- model: `standard_cifar_resnet18`
- architecture style: standard, `spectral_norm=false`, `mod=false`
- optimizer: `sgd`
- epochs: `2`
- batch size: `128`
- seed: `0`
- checkpoint: final checkpoint required, best-validation optional

Expected training outputs:

- `checkpoint_final.pt`
- `train_metrics.jsonl`
- `val_metrics.jsonl`
- `config_snapshot.yaml`
- `git_commit.txt`
- `command.txt`
- `run.log`

Expected evaluator outputs:

- `metrics_classification.json`
- `metrics_calibration.json`
- `metrics_ood_logit.json`
- `metrics_ood_feature.json`
- `metrics_ood_nc_hybrid.json`
- `metrics_geometry.json`
- `detector_params.json`
- `feature_stats.json`

The evaluator must use one shared cache of logits/features. Do not run detectors on separately extracted features.

## Milestone 1 detector minimum

For the smoke run, implement the smallest useful Tier 0 subset:

- logit: `msp`, `energy_id_score`
- feature: `mahalanobis`, `mahalanobis_l2`, `knn`, `gmm_ddu_tied`, `gmm_ddu_diag`, `gmm_ddu_shrinkage`
- geometry: `within_var`, `inter_dist_l2`, `inter_dist_sq`, `nc0_width_norm`, `nc1`, `nc2_mean_cos`, `nc3_self_duality`

Use `reports/METRIC_DEFINITIONS.md` for exact names and score directions.

## Milestone 1 success criteria

The smoke run is complete only when:

1. training writes checkpoint and metadata,
2. cache extraction writes ID train, ID val, ID test, and at least one OOD test cache,
3. evaluator reads that same cache for all detectors,
4. metric JSON files use the revised project metric names,
5. a manifest can be filled from the run outputs,
6. the result copy plan back to local `results/raw/<run_id>/` is clear.

## Milestone 2: standard optimizer and architecture expansion

After Milestone 1 passes:

- add `sgdw`, `adam`, `adamw`, `adam_coupled_decoupled`,
- add `standard_wrn_28_10_dropout03`,
- add `standard_wrn_28_10_nodrop`,
- add SAM-family optimizers only after base optimizer semantics are tested,
- add DDU SN/mod diagnostics only after standard architecture runs are safe.

Required tests:

- optimizer endpoint test: `adam_coupled_decoupled` with ratio `0.0` behaves like AdamW-style decoupled WD,
- optimizer endpoint test: ratio `1.0` behaves like Adam-style coupled WD,
- SN off test: no DDU spectral normalization wrapper in standard main models,
- SN on test: DDU diagnostic model uses DDU SN wrapper,
- mod off test: ReLU and standard shortcut,
- mod on test: LeakyReLU and avg-pool-pad shortcut.

## Milestone 3: first mini sweep

Run the first real mini sweep only after smoke tests pass:

- dataset: CIFAR-10
- model: `standard_cifar_resnet18`
- epochs: `200`
- batch size: `128`
- seeds: `0,1,2`
- optimizers:
  - `sgd`
  - `sgdw`
  - `adam`
  - `adamw`
  - `adam_coupled_decoupled` with coupled ratios `[0.0, 0.25, 0.5, 0.75, 1.0]`

Purpose:

- check whether coupled ratio changes NC/geometry and feature detectors without catastrophic accuracy failure,
- not yet claim the final paper result.

## Local sync after a server run

From Windows PowerShell:

```powershell
$RUN_ID = "YYYYMMDD_HHMM_<server>_<dataset>_<model>_<optimizer>_seed<seed>"
$SERVER = "<user>@<server-host>"
$REMOTE = "~/iclr2027_runs/$RUN_ID"
$LOCAL = "C:\Users\User\Desktop\2027ICLR\results\raw\$RUN_ID"

New-Item -ItemType Directory -Force -Path $LOCAL | Out-Null
scp "$SERVER:$REMOTE/git_commit.txt" $LOCAL
scp "$SERVER:$REMOTE/config.sha256" $LOCAL
scp "$SERVER:$REMOTE/config_snapshot.yaml" $LOCAL
scp "$SERVER:$REMOTE/command.txt" $LOCAL
scp "$SERVER:$REMOTE/run.log" $LOCAL
scp "$SERVER:$REMOTE/metrics*.json" $LOCAL
scp "$SERVER:$REMOTE/detector_params.json" $LOCAL
scp "$SERVER:$REMOTE/feature_stats.json" $LOCAL
```

Then create or update:

```text
results/manifests/<run_id>.json
```

Use `results/manifests/RUN_MANIFEST_TEMPLATE.json` and `ops/RUN_MANIFEST_RULES.md`.

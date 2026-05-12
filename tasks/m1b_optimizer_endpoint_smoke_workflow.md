# Task: M1B Optimizer Endpoint Smoke Workflow

Created: 2026-05-13
Status: completed

## Goal

Add and validate the M1B optimizer endpoint plumbing for:

- `adam`
- `adamw`
- `adam_coupled_decoupled` with `coupled_ratio: 0.0`
- `adam_coupled_decoupled` with `coupled_ratio: 1.0`

The task is complete only if the endpoint optimizer semantics are tested, the 2 epoch smoke train/cache/eval pipeline succeeds, and each run has a manifest.

## Context

Repo path:

```bash
/home/ghjin/2027ICLR/2027ICLR
```

Branch used:

```bash
exp/m1-smoke-pipeline
```

Base commit at execution time:

```text
5c269626f944a307268f441ef523337b352f3d22
```

The M1B runs used working-tree optimizer endpoint changes before those changes were committed. The `101` manifests record this as:

```text
5c269626f944a307268f441ef523337b352f3d22+dirty-m1b-optimizer-endpoints
```

## Files Changed

Optimizer implementation:

- `code/optimizers/adam_coupled_decoupled.py`
- `code/optimizers/factory.py`
- `code/optimizers/__init__.py`

Smoke checks:

- `code/tests/smoke_checks.py`

Smoke configs:

- `configs/smoke/cifar10_standard-cifar-resnet18_adam_2ep_seed0.yaml`
- `configs/smoke/cifar10_standard-cifar-resnet18_adamw_2ep_seed0.yaml`
- `configs/smoke/cifar10_standard-cifar-resnet18_adam-coupled-decoupled_r0_2ep_seed0.yaml`
- `configs/smoke/cifar10_standard-cifar-resnet18_adam-coupled-decoupled_r1_2ep_seed0.yaml`

## Verification

Local code checks:

```bash
python -m py_compile code/optimizers/adam_coupled_decoupled.py code/optimizers/factory.py code/optimizers/__init__.py code/tests/smoke_checks.py code/train.py
python code/tests/smoke_checks.py --config configs/smoke/cifar10_standard-cifar-resnet18_sgd_2ep_seed0.yaml --check optimizer-endpoints
python code/tests/smoke_checks.py --config configs/smoke/cifar10_standard-cifar-resnet18_sgd_2ep_seed0.yaml --check model
```

Config smoke check:

- Each new M1B config parsed successfully.
- Each config passed a random-batch forward/backward/optimizer-step check.

Full 2 epoch pipeline:

```text
train.py -> extract_cache.py -> eval_posthoc.py -> smoke_checks.py --check run-dir
```

This full pipeline was completed for local-curie and SSH host `101`.

## Run Manifests

Local-curie runs:

- `results/manifests/20260512_2334_local_cifar10_standard-cifar-resnet18_adam_seed0.json`
- `results/manifests/20260512_2335_local_cifar10_standard-cifar-resnet18_adamw_seed0.json`
- `results/manifests/20260512_2336_local_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r0_seed0.json`
- `results/manifests/20260512_2337_local_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r1_seed0.json`

SSH host `101` runs:

- `results/manifests/20260512_2356_101_cifar10_standard-cifar-resnet18_adam_seed0.json`
- `results/manifests/20260512_2357_101_cifar10_standard-cifar-resnet18_adamw_seed0.json`
- `results/manifests/20260512_2358_101_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r0_seed0.json`
- `results/manifests/20260512_2359_101_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r1_seed0.json`

Checkpoint/cache policy:

- `checkpoint_final.pt` and cache `.pt` files remain under `/home/ghjin/iclr2027_runs/<run_id>/`.
- `results/raw/<run_id>/` contains only copied metrics/log/config snapshots and is ignored by Git.

## Notes

Confirmed fact:

- The optimizer endpoint implementation is wired into the training pipeline and validated by one-step endpoint tests plus 2 epoch smoke pipelines.

Interpretation:

- The smoke metrics show the pipeline works, not that any optimizer is better for the ICLR claim.

Remaining risks:

- M1B uses 2 epoch smoke configs only.
- The first real sweep still needs matched-protocol 200 epoch configs, seed allocation, and server scheduling.

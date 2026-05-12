# M1 Smoke Status, 2026-05-13

## Purpose

Record the current M1A/M1B smoke validation state before pushing the working changes to `main`.

## Files Checked

Manifests:

- `results/manifests/20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0.json`
- `results/manifests/20260512_2334_local_cifar10_standard-cifar-resnet18_adam_seed0.json`
- `results/manifests/20260512_2335_local_cifar10_standard-cifar-resnet18_adamw_seed0.json`
- `results/manifests/20260512_2336_local_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r0_seed0.json`
- `results/manifests/20260512_2337_local_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r1_seed0.json`
- `results/manifests/20260512_2356_101_cifar10_standard-cifar-resnet18_adam_seed0.json`
- `results/manifests/20260512_2357_101_cifar10_standard-cifar-resnet18_adamw_seed0.json`
- `results/manifests/20260512_2358_101_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r0_seed0.json`
- `results/manifests/20260512_2359_101_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r1_seed0.json`

Metrics files:

- `metrics_classification.json`
- `metrics_geometry.json`
- `metrics_ood_nc_hybrid.json`

Context/contract files:

- `AI_CONTEXT.md`
- `code/IMPLEMENTATION_CONTRACT.md`
- `reports/METRIC_DEFINITIONS.md`
- `results/AGENTS.md`

## Confirmed Metrics

These are 2 epoch smoke validation metrics. They confirm that training, cache extraction, post-hoc evaluation, metric writing, and manifest registration worked.

| run_id | source | optimizer | coupled_ratio | id_test accuracy |
|---|---:|---|---:|---:|
| `20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0` | `101` | `sgd` | n/a | 0.5275999903678894 |
| `20260512_2334_local_cifar10_standard-cifar-resnet18_adam_seed0` | local-curie | `adam` | n/a | 0.7027999758720398 |
| `20260512_2335_local_cifar10_standard-cifar-resnet18_adamw_seed0` | local-curie | `adamw` | n/a | 0.7103000283241272 |
| `20260512_2336_local_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r0_seed0` | local-curie | `adam_coupled_decoupled` | 0.0 | 0.7172999978065491 |
| `20260512_2337_local_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r1_seed0` | local-curie | `adam_coupled_decoupled` | 1.0 | 0.6988999843597412 |
| `20260512_2356_101_cifar10_standard-cifar-resnet18_adam_seed0` | `101` | `adam` | n/a | 0.7027999758720398 |
| `20260512_2357_101_cifar10_standard-cifar-resnet18_adamw_seed0` | `101` | `adamw` | n/a | 0.7103000283241272 |
| `20260512_2358_101_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r0_seed0` | `101` | `adam_coupled_decoupled` | 0.0 | 0.7172999978065491 |
| `20260512_2359_101_cifar10_standard-cifar-resnet18_adam_coupled_decoupled_r1_seed0` | `101` | `adam_coupled_decoupled` | 1.0 | 0.6988999843597412 |

Geometry metric files emitted the revised names:

- `within_var`
- `inter_dist_l2`
- `inter_dist_sq`
- `nc0_width_norm`
- `nc1`
- `nc2_mean_cos`
- `nc3_self_duality`

`metrics_ood_nc_hybrid.json` records that M1A/M1B emits geometry diagnostics only and defers NC/prototype/hybrid detector scores.

## Interpretation

The M1A/M1B smoke pipeline is operational for standard CIFAR ResNet-18, CIFAR-10 ID, SVHN OOD, shared-cache evaluation, and the first optimizer endpoint axis.

The `adam_coupled_decoupled` implementation has endpoint tests:

- `coupled_ratio=0.0` matches PyTorch AdamW one-step behavior.
- `coupled_ratio=1.0` matches PyTorch Adam one-step behavior.

## Risks Or Missing Evidence

- These are 2 epoch smoke runs, not paper evidence.
- No multi-seed or 200 epoch matched-protocol sweep has been run.
- The M1B `101` runs were executed from a dirty working tree before the optimizer endpoint implementation was committed; manifests record this explicitly.
- Checkpoints and cache `.pt` files are preserved in `/home/ghjin/iclr2027_runs/<run_id>/` and are not Git-tracked.

## Next Actions

- Commit and push M1A/M1B code, configs, task notes, report, and run manifests.
- Prepare matched-protocol 200 epoch configs only after deciding the first sweep scope and server allocation.
- Continue to separate confirmed metrics, interpretation, and ICLR hypotheses in future reports.

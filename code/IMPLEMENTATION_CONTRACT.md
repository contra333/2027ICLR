# ICLR 2027 Code Implementation Contract

Last updated: 2026-05-12 KST

## Core decision

The server code must be an ICLR 2027 core codebase, not a continuation of the old DDU fork.

Implement:

- standard training,
- explicit optimizer semantics,
- explicit model feature return,
- shared-cache post-hoc evaluator,
- traceable run outputs.

Use `DDU_fork` only as a diagnostic reference for DDU spectral normalization, mod settings, legacy SAM mechanics, and DDU/GMM feature-density behavior.

## Required code layout

Start with this minimal structure under `code/`:

```text
code/
  train.py
  extract_cache.py
  eval_posthoc.py
  models/
  optimizers/
  train_utils/
  eval/
  tests/
```

Do not build every future method at once. Milestone 1 should be small enough to run and debug quickly on one server.

## Model API

All models must support:

```python
logits, features = model(x, return_features=True)
```

Rules:

- `features` is the flattened penultimate feature before the classifier.
- `logits` is the classifier output before softmax.
- Do not rely on `self.feature` side effects for evaluator behavior.
- Record `feature_layer` in config and metadata.
- Main models must default to `spectral_norm=false` and `mod=false`.

## Architecture source of truth

Use `code/models/ARCHITECTURE_CONTRACT.md` for model definitions.

Main ICLR evidence uses:

- `standard_cifar_resnet18`
- `standard_wrn_28_10_dropout03`
- `standard_wrn_28_10_nodrop` only as the dropout ablation

DDU-style SN/mod models are diagnostic only.

## Optimizer factory

Create an optimizer factory instead of embedding optimizer decisions in `train.py`.

Required names:

- `sgd`: PyTorch SGD with coupled weight decay
- `sgdw`: decoupled SGD weight decay
- `adam`: Adam with coupled weight decay
- `adamw`: AdamW with decoupled weight decay
- `adam_coupled_decoupled`: Adam-to-AdamW interpolation
- `sam_sgd`: SAM with SGD base optimizer
- `asam_sgd`: adaptive SAM with SGD base optimizer
- `gsam_sgd`: GSAM with SGD base optimizer

Milestone 1 only requires `sgd`, `adam`, `adamw`, and `adam_coupled_decoupled`.

## Adam-to-AdamW interpolation

Use this semantic contract:

```text
total_weight_decay = 5e-4
coupled_ratio = r
wd_coupled = r * total_weight_decay
wd_decoupled = (1 - r) * total_weight_decay

r = 0.0 -> AdamW-style decoupled WD
r = 1.0 -> Adam-style coupled WD
```

Update rule:

```text
g_eff = grad + wd_coupled * p_old
m = beta1 * m + (1 - beta1) * g_eff
v = beta2 * v + (1 - beta2) * g_eff^2
p_decayed = p_old * (1 - lr * wd_decoupled)
p_new = p_decayed - lr * m_hat / (sqrt(v_hat) + eps)
```

Required tests:

- one-step toy test against AdamW-style behavior at `r=0.0`,
- one-step toy test against Adam-style behavior at `r=1.0`,
- record any intentional numerical difference from PyTorch defaults.

## Weight decay policy

Expose weight decay policy in config:

- `all_params`: legacy/DDU diagnostic only,
- `weights_only_no_bias_norm`: main CNN default,
- `vit_no_ln_bias`: ViT extension default.

For main CNN runs, apply weight decay to convolution and linear weights, including classifier weights. Exclude bias, BatchNorm, and LayerNorm parameters.

Do not silently change this policy between runs.

## Training protocol

Keep these protocols separate:

- `matched_protocol`: same dataset, architecture, augmentation, epoch budget, schedule, seed set, and total WD; change only the intended optimizer axis.
- `tuned_protocol`: optimizer LR/WD grid is predeclared and selected by ID validation accuracy, NLL, or ECE only.

Never tune optimizer or detector hyperparameters using OOD test AUROC.

Default first budgets:

- smoke: 2 epochs,
- first sweep: 200 epochs, milestones at 1/3 and 2/3, seeds `0,1,2`,
- NeurIPS bridge: 350 epochs, milestones `150` and `250`.

## Training outputs

Each training run should write:

- `checkpoint_final.pt`
- `checkpoint_best_val.pt` if best-validation checkpoint is enabled
- `train_metrics.jsonl`
- `val_metrics.jsonl`
- `config_snapshot.yaml`
- `git_commit.txt`
- `command.txt`
- `run.log`

The final checkpoint is the main checkpoint for geometry and detector analysis.

## Shared-cache evaluator

Implement evaluation in this order:

1. `extract_cache.py` extracts logits, features, labels, classifier weights, and metadata once.
2. `eval_posthoc.py` runs all post-hoc detectors on that cache.
3. The evaluator writes split metric files following `reports/METRIC_DEFINITIONS.md`.

Required cache splits:

- `id_train`
- `id_val`
- `id_test`
- `ood_test/<ood_name>`

All project OOD detector scores must be higher = ID-like.

## Milestone 1 evaluator subset

Implement first:

- classification: `accuracy`, `nll`
- calibration: `ece_15bin`, `temperature_scaled_ece_15bin`
- logit: `msp`, `energy_id_score`
- feature: `mahalanobis`, `mahalanobis_l2`, `knn`, `gmm_ddu_tied`, `gmm_ddu_diag`, `gmm_ddu_shrinkage`
- geometry: `within_var`, `inter_dist_l2`, `inter_dist_sq`, `nc0_width_norm`, `nc1`, `nc2_mean_cos`, `nc3_self_duality`

Do not emit legacy names such as `nc0`, `nc3`, `nc4`, or `inter_dist`.

## Required smoke tests

Before any 200 epoch run:

- model forward test for `standard_cifar_resnet18`,
- feature shape test for `return_features=True`,
- optimizer endpoint tests for `adam_coupled_decoupled`,
- SN/mod off tests for main models,
- minimal train step test,
- minimal cache extraction test,
- evaluator JSON parse test.

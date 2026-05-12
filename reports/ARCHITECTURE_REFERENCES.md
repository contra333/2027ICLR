# Architecture and Optimizer References for ICLR 2027

Last updated: 2026-05-12 KST

## Purpose

This file gives server-side Codex, Codex CLI, and human implementers the reference hierarchy for architecture and optimizer implementation.

Use this file with:

- `code/models/ARCHITECTURE_CONTRACT.md`
- `code/IMPLEMENTATION_CONTRACT.md`
- `reports/METRIC_DEFINITIONS.md`

## Project-level rule

Main ICLR evidence uses standard CIFAR CNNs:

- `standard_cifar_resnet18`
- `standard_wrn_28_10_dropout03`

Do not use DDU-specific spectral normalization or modified activations for the main claim. DDU-style SN/mod settings are diagnostic or appendix only.

## CIFAR ResNet-18 reference priority

1. He et al., Deep Residual Learning for Image Recognition
   https://arxiv.org/abs/1512.03385
   Use for the conceptual residual learning framework and BasicBlock motivation.

2. MMPretrain `ResNet_CIFAR`
   https://onedl-mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.backbones.ResNet_CIFAR.html
   Use for the CIFAR stem rule: `conv1` uses kernel size `3`, stride `1`, and no maxpool after the stem.

3. Torchvision ResNet source docs
   https://brianjo.github.io/vision/stable/_modules/torchvision/models/resnet.html
   Use for BasicBlock semantics: `conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> residual add -> ReLU`, with downsample when stride or channel count changes.

4. `kuangliu/pytorch-cifar`
   https://github.com/kuangliu/pytorch-cifar
   Use only as CIFAR implementation sanity check, not as the formal project contract.

Implementation rule:

- Do not use unchanged `torchvision.models.resnet18` for CIFAR main runs.
- Audit phrase: `torchvision.models.resnet18 unchanged` is not an acceptable CIFAR main architecture.
- Implement `standard_cifar_resnet18` exactly as defined in `code/models/ARCHITECTURE_CONTRACT.md`.

## WRN-28-10 reference priority

1. Zagoruyko and Komodakis, Wide Residual Networks
   https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html
   Use as the paper-level reference for decreasing depth and increasing width in WRNs.

2. Official WRN repository
   https://github.com/szagoruyko/wide-residual-networks
   Use for WRN-28-10 and WRN-28-10-dropout reference behavior. The repository reports both WRN-28-10 and WRN-28-10-dropout on CIFAR and includes example usage with `depth=28`, `widen_factor=10`, and `dropout=0.3`.

3. Local `DDU_fork-main/net/wide_resnet.py`
   Use only as local diagnostic reference for old DDU-style settings. Do not use it as the main architecture source unless SN and mod behavior are explicitly controlled.

Implementation rule:

- Use `standard_wrn_28_10_dropout03` for the main WRN bridge.
- Use `standard_wrn_28_10_nodrop` for the geometry-clean dropout ablation.
- Do not silently switch dropout rates between runs.

## DDU diagnostic reference

Local reference path:

```text
C:\Users\User\Desktop\ICML+통계학회(연구)\DDU_fork-main
```

Observed local status:

- This folder exists locally.
- It is not currently a Git repository.
- It should not be treated as a required server clone unless a remote URL is provided later.

Use DDU_fork only for:

- DDU spectral normalization wrapper reference,
- DDU `mod=true` architecture reference,
- old NeurIPS/DDU bridge diagnostics,
- legacy SAM two-step reference,
- legacy GMM/DDU feature-density reference.

Do not use DDU_fork as main architecture source.

## Optimizer semantics references

PyTorch Adam:

- https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html
- Use for coupled weight decay semantics where weight decay enters the Adam gradient/moment computation.

PyTorch AdamW:

- https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html
- Use for decoupled weight decay semantics where weight decay does not accumulate in momentum or variance.

Decoupled Weight Decay Regularization:

- https://arxiv.org/abs/1711.05101
- Use as the paper-level reference for separating L2 regularization and decoupled weight decay, especially for adaptive optimizers.

Project-specific optimizer rule:

- `adam_coupled_decoupled` is a controlled interpolation axis, not a generic optimizer recommendation.
- Total weight decay stays fixed while coupled ratio changes.

## SAM-family references

SAM local reference:

- `C:\Users\User\Desktop\ICML+통계학회(연구)\DDU_fork-main\utils\sam.py`
- Use as a local two-step implementation reference only.

ASAM:

- https://arxiv.org/abs/2102.11600
- Use for adaptive sharpness-aware minimization motivation and scale-invariant perturbation behavior.

GSAM:

- https://arxiv.org/abs/2203.08065
- Use for surrogate-gap guided sharpness-aware minimization.

Project-specific SAM rule:

- SAM-family comparisons are protocol-matched, not compute-equivalent.
- Record `rho`, base optimizer, base LR, momentum, weight decay, and schedule for every SAM-family run.

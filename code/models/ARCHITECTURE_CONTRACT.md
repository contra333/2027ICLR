# Architecture Contract

Last updated: 2026-05-12 KST

## Project-level rule

Main ICLR evidence uses standard CIFAR CNNs, not DDU-specific architectures.

Main settings:

```yaml
architecture_style: standard
spectral_norm:
  enabled: false
mod:
  enabled: false
```

DDU-style spectral normalization and mod settings are allowed only for appendix or diagnostic runs.

## `standard_cifar_resnet18`

Use a CIFAR-adapted ResNet-18, not unchanged `torchvision.models.resnet18`.
Search phrase for audits: `torchvision.models.resnet18 unchanged` is forbidden for CIFAR main experiments.

Contract:

- Input: `3 x 32 x 32`.
- Stem: `Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=false)`, `BatchNorm2d(64)`, ReLU.
- No initial `7 x 7` convolution.
- No initial maxpool.
- Block: standard ResNet BasicBlock.
- BasicBlock order: `conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> residual add -> ReLU`.
- Stage channels: `[64, 128, 256, 512]`.
- Blocks per stage: `[2, 2, 2, 2]`.
- Stage strides: `[1, 2, 2, 2]`.
- Shortcut: identity when possible; `1 x 1` conv plus BN when stride or channel count changes.
- Pool: `AdaptiveAvgPool2d(1)`.
- Classifier: `Linear(512, num_classes)`.
- Feature: flattened pooled vector before classifier.
- Main setting: `spectral_norm=false`, `mod=false`, `ddu_style=false`.
- Forward API: `logits, features = model(x, return_features=True)`.

## `standard_wrn_28_10_dropout03`

Use the Wide Residual Networks WRN-28-10 definition.

Contract:

- Input: `3 x 32 x 32`.
- Depth: `28`.
- Widen factor: `10`.
- Blocks per group: `(28 - 4) / 6 = 4`.
- Stem: `Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=false)`.
- Group channels: `[160, 320, 640]`.
- Group strides: `[1, 2, 2]`.
- Block: `BN -> ReLU -> Conv -> BN -> ReLU -> Dropout -> Conv`.
- Shortcut: identity when possible; `1 x 1` conv when stride or channel count changes.
- Dropout: `0.3`.
- Pool: global average pooling.
- Classifier: `Linear(640, num_classes)`.
- Feature: flattened pooled vector before classifier.
- Main setting: `spectral_norm=false`, `mod=false`, `ddu_style=false`.
- Forward API: `logits, features = model(x, return_features=True)`.

Use this as the WRN bridge setting because it is closest to common WRN-28-10-dropout usage and the old DDU/NeurIPS bridge setting.

## `standard_wrn_28_10_nodrop`

Same as `standard_wrn_28_10_dropout03`, except:

- Dropout: `0.0`.

Use this as a geometry-clean WRN ablation when dropout is a concern.

## `ddu_diagnostic` variants

DDU diagnostic variants are not main evidence.

Diagnostic contract:

```yaml
architecture_style: ddu_diagnostic
spectral_norm:
  enabled: true
  backend: ddu
  coeff: 3.0
  n_power_iterations: 1
mod:
  enabled: true
  activation: leaky_relu
  shortcut: avg_pool_pad
```

Use these settings only to bridge old NeurIPS/DDU experiments or to test whether SN/mod architecture choices confound optimizer effects.

## Forbidden for main experiments

- Do not use unchanged `torchvision.models.resnet18` for CIFAR main runs.
- Do not use DDU_fork defaults as main architecture defaults.
- Do not enable DDU spectral normalization in standard main configs.
- Do not enable DDU `mod=true` in standard main configs.
- Do not rely on `self.feature` side effects.

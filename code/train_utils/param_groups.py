from __future__ import annotations

from torch import nn


NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def make_weight_decay_param_groups(
    model: nn.Module, weight_decay: float, policy: str
) -> list[dict]:
    if policy == "all_params":
        return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]
    if policy != "weights_only_no_bias_norm":
        raise ValueError(f"Unsupported weight_decay_policy for Milestone 1A: {policy}")

    decay = []
    no_decay = []
    seen = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if full_name in seen:
                continue
            seen.add(full_name)
            if param_name.endswith("bias"):
                no_decay.append(param)
            elif isinstance(module, NORM_TYPES):
                no_decay.append(param)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                decay.append(param)
            elif param.ndim <= 1:
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

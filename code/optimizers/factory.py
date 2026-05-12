from __future__ import annotations

from typing import Any

from torch import nn
from torch.optim import SGD

from train_utils.param_groups import make_weight_decay_param_groups


def build_optimizer(model: nn.Module, cfg: dict[str, Any]):
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name")
    if name != "sgd":
        raise ValueError(f"Milestone 1A implements only optimizer.name=sgd, got {name}")

    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    policy = opt_cfg.get("weight_decay_policy", "weights_only_no_bias_norm")
    param_groups = make_weight_decay_param_groups(model, weight_decay, policy)
    return SGD(
        param_groups,
        lr=float(opt_cfg["lr"]),
        momentum=float(opt_cfg.get("momentum", 0.0)),
        nesterov=bool(opt_cfg.get("nesterov", False)),
    )

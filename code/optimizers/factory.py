from __future__ import annotations

from typing import Any

from torch import nn
from torch.optim import Adam, AdamW, SGD

from optimizers.adam_coupled_decoupled import AdamCoupledDecoupled
from train_utils.param_groups import make_weight_decay_param_groups


def _adam_betas(opt_cfg: dict[str, Any]) -> tuple[float, float]:
    if "betas" in opt_cfg:
        beta1, beta2 = opt_cfg["betas"]
        return float(beta1), float(beta2)
    return float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.999))


def build_optimizer(model: nn.Module, cfg: dict[str, Any]):
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name")

    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    policy = opt_cfg.get("weight_decay_policy", "weights_only_no_bias_norm")
    param_groups = make_weight_decay_param_groups(model, weight_decay, policy)
    lr = float(opt_cfg["lr"])

    if name == "sgd":
        return SGD(
            param_groups,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.0)),
            nesterov=bool(opt_cfg.get("nesterov", False)),
        )
    if name == "adam":
        return Adam(
            param_groups,
            lr=lr,
            betas=_adam_betas(opt_cfg),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )
    if name == "adamw":
        return AdamW(
            param_groups,
            lr=lr,
            betas=_adam_betas(opt_cfg),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )
    if name == "adam_coupled_decoupled":
        return AdamCoupledDecoupled(
            param_groups,
            lr=lr,
            betas=_adam_betas(opt_cfg),
            eps=float(opt_cfg.get("eps", 1e-8)),
            coupled_ratio=float(opt_cfg.get("coupled_ratio", 0.5)),
        )

    raise ValueError(f"Unsupported optimizer.name={name}")

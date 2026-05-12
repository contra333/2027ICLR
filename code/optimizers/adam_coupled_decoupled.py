from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamCoupledDecoupled(Optimizer):
    """Adam with a fixed split between coupled and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        coupled_ratio: float = 0.5,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= coupled_ratio <= 1.0:
            raise ValueError(f"Invalid coupled_ratio: {coupled_ratio}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "coupled_ratio": coupled_ratio,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            total_weight_decay = group["weight_decay"]
            coupled_ratio = group["coupled_ratio"]
            wd_coupled = coupled_ratio * total_weight_decay
            wd_decoupled = (1.0 - coupled_ratio) * total_weight_decay

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamCoupledDecoupled does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                effective_grad = grad
                if wd_coupled != 0.0:
                    effective_grad = grad.add(param, alpha=wd_coupled)

                exp_avg.mul_(beta1).add_(effective_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    effective_grad, effective_grad, value=1.0 - beta2
                )

                if wd_decoupled != 0.0:
                    param.mul_(1.0 - lr * wd_decoupled)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

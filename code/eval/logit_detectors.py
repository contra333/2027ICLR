from __future__ import annotations

import torch
import torch.nn.functional as F


def msp(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).max(dim=1).values


def maxlogit(logits: torch.Tensor) -> torch.Tensor:
    return logits.max(dim=1).values


def energy_id_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def neg_entropy(logits: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return (probs * torch.log(probs.clamp_min(eps))).sum(dim=1)


LOGIT_DETECTOR_REGISTRY = {
    "msp": msp,
    "maxlogit": maxlogit,
    "energy_id_score": energy_id_score,
    "neg_entropy": neg_entropy,
}

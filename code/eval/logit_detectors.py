from __future__ import annotations

import torch
import torch.nn.functional as F


def msp(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).max(dim=1).values


def energy_id_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return temperature * torch.logsumexp(logits / temperature, dim=1)

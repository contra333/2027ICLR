from __future__ import annotations

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels, reduction="mean").item())


def ece_15bin(logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    probs = F.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    correct = (preds == labels).float()
    ece = torch.zeros((), dtype=torch.float64)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=logits.device)
    for i in range(num_bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if i == 0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf > lower) & (conf <= upper)
        if mask.any():
            bin_weight = mask.float().mean().double()
            bin_acc = correct[mask].mean().double()
            bin_conf = conf[mask].mean().double()
            ece += bin_weight * torch.abs(bin_acc - bin_conf)
    return float(ece.item())


def fit_temperature(id_val_logits: torch.Tensor, id_val_labels: torch.Tensor) -> float:
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    logits = id_val_logits.double()
    labels = id_val_labels.long()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        temperature = float(torch.exp(log_temperature.detach()).clamp(min=1e-3, max=100.0).item())
    except Exception:
        temperature = 1.0
    return temperature

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    loss = F.cross_entropy(logits, labels, reduction="mean")
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean()
    return {"accuracy": float(acc.item()), "nll": float(loss.item())}


def update_running(total: dict[str, float], batch_metrics: dict[str, float], count: int) -> None:
    total["num_samples"] = total.get("num_samples", 0.0) + count
    for key, value in batch_metrics.items():
        total[key] = total.get(key, 0.0) + float(value) * count


def finalize_running(total: dict[str, float]) -> dict[str, float]:
    num_samples = int(total.get("num_samples", 0))
    if num_samples == 0:
        return {"num_samples": 0, "accuracy": 0.0, "nll": 0.0}
    return {
        "num_samples": num_samples,
        "accuracy": total.get("accuracy", 0.0) / num_samples,
        "nll": total.get("nll", 0.0) / num_samples,
    }

from __future__ import annotations

from typing import Any

from .cifar_resnet import standard_cifar_resnet18


def build_model(cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    name = model_cfg.get("name")
    if name != "standard_cifar_resnet18":
        raise ValueError(f"Milestone 1A supports only standard_cifar_resnet18, got {name}")
    if model_cfg.get("architecture_style") != "standard":
        raise ValueError("Milestone 1A requires model.architecture_style=standard")
    if model_cfg.get("spectral_norm", {}).get("enabled", False):
        raise ValueError("Milestone 1A main model must use spectral_norm.enabled=false")
    if model_cfg.get("mod", {}).get("enabled", False):
        raise ValueError("Milestone 1A main model must use mod.enabled=false")
    return standard_cifar_resnet18(num_classes=int(model_cfg.get("num_classes", 10)))

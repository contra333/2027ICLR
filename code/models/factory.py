from __future__ import annotations

from typing import Any

from .cifar_resnet import standard_cifar_resnet18
from .wide_resnet import standard_wrn_28_10_dropout03, standard_wrn_28_10_nodrop


MODEL_REGISTRY = {
    "standard_cifar_resnet18": standard_cifar_resnet18,
    "standard_wrn_28_10_dropout03": standard_wrn_28_10_dropout03,
    "standard_wrn_28_10_nodrop": standard_wrn_28_10_nodrop,
}


def build_model(cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    name = str(model_cfg.get("name"))
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    if model_cfg.get("architecture_style") != "standard":
        raise ValueError("Main ICLR models require model.architecture_style=standard")
    if model_cfg.get("spectral_norm", {}).get("enabled", False):
        raise ValueError("Main ICLR models must use spectral_norm.enabled=false")
    if model_cfg.get("mod", {}).get("enabled", False):
        raise ValueError("Main ICLR models must use mod.enabled=false")
    return MODEL_REGISTRY[name](num_classes=int(model_cfg.get("num_classes", 10)))


def get_classifier_state(model) -> dict[str, Any]:
    classifier = getattr(model, "fc", None)
    if classifier is None:
        classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        raise ValueError(
            f"Cannot find classifier weight on model type {type(model).__name__}; "
            "expected .fc or .classifier"
        )
    bias = getattr(classifier, "bias", None)
    return {
        "weight": classifier.weight.detach().cpu(),
        "bias": bias.detach().cpu() if bias is not None else None,
        "orientation": "rows_are_classes",
        "classifier_attr": "fc" if getattr(model, "fc", None) is classifier else "classifier",
    }

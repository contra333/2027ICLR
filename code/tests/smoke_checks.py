from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data import DATASET_REGISTRY, build_data_bundle  # noqa: E402
from data.registry import get_dataset_spec  # noqa: E402
from eval.aggregate_metrics import fpr95  # noqa: E402
from eval.feature_detectors import fit_knn, l2_normalize, score_knn  # noqa: E402
from eval.geometry import compute_geometry  # noqa: E402
from eval.logit_detectors import energy_id_score, maxlogit, msp, neg_entropy  # noqa: E402
from eval_posthoc import _validate_names  # noqa: E402
from models import MODEL_REGISTRY, build_model, get_classifier_state  # noqa: E402
from optimizers import AdamCoupledDecoupled, build_optimizer  # noqa: E402
from train_utils import load_config, set_seed  # noqa: E402


EXPECTED_EVAL_JSON = [
    "metrics_classification.json",
    "metrics_calibration.json",
    "metrics_ood_logit.json",
    "metrics_ood_feature.json",
    "metrics_ood_nc_hybrid.json",
    "metrics_geometry.json",
    "detector_params.json",
    "feature_stats.json",
]


def check_model(cfg):
    model = build_model(cfg)
    model.eval()
    expected_classes = int(cfg["model"].get("num_classes", 10))
    image_size = int(cfg.get("data", {}).get("image_size", 32))
    x = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        logits, features = model(x, return_features=True)
    assert logits.shape == (2, expected_classes), logits.shape
    assert features.ndim == 2, features.shape
    if hasattr(model, "feature_dim"):
        assert features.shape[1] == int(model.feature_dim), features.shape
    classifier = get_classifier_state(model)
    assert classifier["weight"].shape == (expected_classes, features.shape[1])
    assert not any("spectral" in type(module).__name__.lower() for module in model.modules())


def check_data(cfg):
    set_seed(int(cfg.get("run", {}).get("seed", 0)))
    data = build_data_bundle(cfg)
    total = int(data.split_metadata["num_total_train"])
    val_fraction = float(data.split_metadata["val_fraction"])
    expected_val = int(round(total * val_fraction))
    assert data.split_metadata["num_train"] == total - expected_val, data.split_metadata
    assert data.split_metadata["num_val"] == expected_val, data.split_metadata
    requested_ood = {
        entry.lower() if isinstance(entry, str) else str(entry.get("name", entry.get("dataset"))).lower()
        for entry in cfg.get("data", {}).get("ood_datasets", [])
    }
    assert requested_ood.issubset(set(data.ood_test_loaders)), data.ood_test_loaders.keys()


def check_train_step(cfg):
    set_seed(int(cfg.get("run", {}).get("seed", 0)))
    data = build_data_bundle(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    model.train()
    images, labels, _ = next(iter(data.train_aug_loader))
    logits, features = model(images, return_features=True)
    assert logits.shape[0] == labels.shape[0]
    assert features.ndim == 2
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _clone_optimizer_params():
    return [
        nn.Parameter(torch.tensor([1.5, -0.5, 0.25], dtype=torch.float64)),
        nn.Parameter(torch.tensor([-1.0, 0.75], dtype=torch.float64)),
    ]


def _assign_toy_grads(params):
    grads = [
        torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64),
        torch.tensor([-0.03, 0.07], dtype=torch.float64),
    ]
    for param, grad in zip(params, grads, strict=True):
        param.grad = grad.clone()


def _toy_param_groups(params, weight_decay):
    return [
        {"params": [params[0]], "weight_decay": weight_decay},
        {"params": [params[1]], "weight_decay": 0.0},
    ]


def _one_step_params(optimizer_cls, coupled_ratio=None):
    params = _clone_optimizer_params()
    _assign_toy_grads(params)
    kwargs = {
        "lr": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    if coupled_ratio is not None:
        kwargs["coupled_ratio"] = coupled_ratio
    optimizer = optimizer_cls(_toy_param_groups(params, weight_decay=0.1), **kwargs)
    optimizer.step()
    return [param.detach().clone() for param in params]


def check_optimizer_endpoints(_cfg):
    adamw_ref = _one_step_params(torch.optim.AdamW)
    r0_actual = _one_step_params(AdamCoupledDecoupled, coupled_ratio=0.0)
    for expected, actual in zip(adamw_ref, r0_actual, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    adam_ref = _one_step_params(torch.optim.Adam)
    r1_actual = _one_step_params(AdamCoupledDecoupled, coupled_ratio=1.0)
    for expected, actual in zip(adam_ref, r1_actual, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def check_unit_metrics(_cfg):
    logits = torch.tensor([[1.0, 3.0, -1.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(msp(logits), torch.softmax(logits, dim=1).max(dim=1).values)
    torch.testing.assert_close(maxlogit(logits), torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(energy_id_score(logits), torch.logsumexp(logits, dim=1))
    expected_neg_entropy = (torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1)).sum(dim=1)
    torch.testing.assert_close(neg_entropy(logits), expected_neg_entropy)

    id_scores = np.arange(20, dtype=np.float64)
    ood_scores = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    assert fpr95(id_scores, ood_scores) == 2.0 / 3.0
    assert fpr95([0.0] * 6 + [1.0] * 94, [-1.0, 0.0, 1.0]) == 2.0 / 3.0

    train = l2_normalize(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    query = l2_normalize(torch.tensor([[4.0, 0.0], [1.0, 1.0]]))
    scores = score_knn(fit_knn(train, k=1), query)
    assert scores[0] > scores[1], scores

    sqrt3 = math.sqrt(3.0)
    simplex = torch.tensor([[1.0, 0.0], [-0.5, sqrt3 / 2.0], [-0.5, -sqrt3 / 2.0]])
    labels = torch.tensor([0, 1, 2])
    metrics = compute_geometry(simplex, labels, simplex)
    assert abs(metrics["inter_dist_l2"] - sqrt3) < 1.0e-6, metrics["inter_dist_l2"]
    assert metrics["nc0_width_norm"] < 1.0e-12, metrics["nc0_width_norm"]
    assert metrics["nc2_mean_etf"] < 1.0e-8, metrics["nc2_mean_etf"]
    assert metrics["nc3_self_duality_raw"] < 1.0e-12, metrics["nc3_self_duality_raw"]


def check_registries(cfg):
    assert "standard_cifar_resnet18" in MODEL_REGISTRY
    assert "standard_wrn_28_10_dropout03" in MODEL_REGISTRY
    assert "standard_wrn_28_10_nodrop" in MODEL_REGISTRY
    assert "cifar10" in DATASET_REGISTRY
    assert "cifar100" in DATASET_REGISTRY
    assert "svhn" in DATASET_REGISTRY
    assert "mnist" in DATASET_REGISTRY
    assert "tiny_imagenet" in DATASET_REGISTRY

    bad_model_cfg = copy.deepcopy(cfg)
    bad_model_cfg["model"]["name"] = "missing_model"
    try:
        build_model(bad_model_cfg)
    except ValueError as exc:
        assert "Unknown model name" in str(exc)
    else:
        raise AssertionError("unknown model name did not raise")

    try:
        get_dataset_spec("missing_dataset")
    except ValueError as exc:
        assert "Unknown dataset name" in str(exc)
    else:
        raise AssertionError("unknown dataset name did not raise")

    try:
        _validate_names(["missing_detector"], {"known": object()}, "test")
    except ValueError as exc:
        assert "Unknown test" in str(exc)
    else:
        raise AssertionError("unknown detector name did not raise")


def _final_cache_dir(run_dir: Path) -> Path:
    tagged = run_dir / "cache" / "final"
    if tagged.exists():
        return tagged
    return run_dir / "cache"


def check_run_dir(run_dir: Path):
    checkpoint = run_dir / "checkpoints" / "checkpoint_final.pt"
    assert checkpoint.exists(), f"missing {checkpoint}"
    for rel in ["train_metrics.jsonl", "val_metrics.jsonl"]:
        path = run_dir / rel
        assert path.exists(), f"missing {path}"

    cache_dir = _final_cache_dir(run_dir)
    cache_metadata_path = cache_dir / "cache_metadata.json"
    required_cache = ["id_train.pt", "id_val.pt", "id_test.pt", "classifier.pt", "cache_metadata.json"]
    if cache_metadata_path.exists():
        metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        required_cache.extend(f"ood_test_{name}.pt" for name in metadata.get("ood_datasets", []))
    else:
        required_cache.append("ood_test_svhn.pt")
    for filename in required_cache:
        path = cache_dir / filename
        assert path.exists(), f"missing {path}"

    for filename in EXPECTED_EVAL_JSON:
        path = run_dir / filename
        assert path.exists(), f"missing {path}"
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    metrics_geometry = json.loads((run_dir / "metrics_geometry.json").read_text(encoding="utf-8"))
    emitted = set(metrics_geometry["id_train"].keys())
    forbidden = {"nc0", "nc3", "nc4", "inter_dist"}
    assert not emitted.intersection(forbidden), emitted.intersection(forbidden)
    required = {
        "within_var",
        "inter_dist_l2",
        "inter_dist_sq",
        "nc0_width_norm",
        "nc1",
        "nc2_mean_cos",
        "nc3_self_duality",
    }
    assert required.issubset(emitted), required - emitted


def main():
    parser = argparse.ArgumentParser(description="M1 smoke checks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--check",
        choices=[
            "model",
            "data",
            "train-step",
            "optimizer-endpoints",
            "unit-metrics",
            "registry",
            "run-dir",
            "all",
        ],
        default="model",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.check in ("model", "all"):
        check_model(cfg)
    if args.check in ("data", "all"):
        check_data(cfg)
    if args.check in ("train-step", "all"):
        check_train_step(cfg)
    if args.check in ("optimizer-endpoints", "all"):
        check_optimizer_endpoints(cfg)
    if args.check in ("unit-metrics", "all"):
        check_unit_metrics(cfg)
    if args.check in ("registry", "all"):
        check_registries(cfg)
    if args.check in ("run-dir", "all"):
        if not args.run_dir:
            raise SystemExit("--run-dir is required for run-dir/all checks")
        check_run_dir(Path(args.run_dir))


if __name__ == "__main__":
    main()

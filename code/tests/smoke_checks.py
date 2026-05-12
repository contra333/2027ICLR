from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data import build_data_bundle  # noqa: E402
from models import build_model  # noqa: E402
from optimizers import build_optimizer  # noqa: E402
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

EXPECTED_CACHE = [
    "cache/id_train.pt",
    "cache/id_val.pt",
    "cache/id_test.pt",
    "cache/ood_test_svhn.pt",
    "cache/classifier.pt",
    "cache/cache_metadata.json",
]


def check_model(cfg):
    model = build_model(cfg)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits, features = model(x, return_features=True)
    assert logits.shape == (2, 10), logits.shape
    assert features.shape == (2, 512), features.shape
    assert not any("spectral" in type(module).__name__.lower() for module in model.modules())


def check_data(cfg):
    set_seed(int(cfg.get("run", {}).get("seed", 0)))
    data = build_data_bundle(cfg)
    assert data.split_metadata["num_train"] == 45000, data.split_metadata
    assert data.split_metadata["num_val"] == 5000, data.split_metadata
    assert "svhn" in data.ood_test_loaders


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


def check_run_dir(run_dir: Path):
    for rel in ["checkpoint_final.pt", "train_metrics.jsonl", "val_metrics.jsonl", *EXPECTED_CACHE]:
        path = run_dir / rel
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
    parser = argparse.ArgumentParser(description="M1A smoke checks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--check",
        choices=["model", "data", "train-step", "run-dir", "all"],
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
    if args.check in ("run-dir", "all"):
        if not args.run_dir:
            raise SystemExit("--run-dir is required for run-dir/all checks")
        check_run_dir(Path(args.run_dir))


if __name__ == "__main__":
    main()

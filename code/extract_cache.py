from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import build_data_bundle
from models import build_model
from train_utils import load_config, set_seed


def _unpack_batch(batch):
    if len(batch) == 3:
        images, labels, indices = batch
        return images, labels, indices
    images, labels = batch
    return images, labels, None


@torch.no_grad()
def extract_split(model, loader, device, split_name: str) -> dict[str, torch.Tensor | str]:
    model.eval()
    logits_list = []
    features_list = []
    labels_list = []
    indices_list = []
    for batch in loader:
        images, labels, indices = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        logits, features = model(images, return_features=True)
        logits_list.append(logits.cpu())
        features_list.append(features.cpu())
        labels_list.append(labels.cpu())
        if indices is not None:
            indices_list.append(indices.cpu())
    record: dict[str, torch.Tensor | str] = {
        "split": split_name,
        "logits": torch.cat(logits_list, dim=0),
        "features": torch.cat(features_list, dim=0),
        "labels": torch.cat(labels_list, dim=0).long(),
    }
    if indices_list:
        record["indices"] = torch.cat(indices_list, dim=0).long()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract shared logits/features cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("run", {}).get("seed", 0))
    set_seed(seed)
    run_dir = Path(args.run_dir)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = build_data_bundle(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(run_dir / "checkpoint_final.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    splits = {
        "id_train": data.id_train_fit_loader,
        "id_val": data.id_val_loader,
        "id_test": data.id_test_loader,
    }
    for split_name, loader in splits.items():
        torch.save(extract_split(model, loader, device, split_name), cache_dir / f"{split_name}.pt")

    for ood_name, loader in data.ood_test_loaders.items():
        split_name = f"ood_test_{ood_name}"
        torch.save(extract_split(model, loader, device, split_name), cache_dir / f"{split_name}.pt")

    classifier = {
        "weight": model.fc.weight.detach().cpu(),
        "bias": model.fc.bias.detach().cpu() if model.fc.bias is not None else None,
        "orientation": "rows_are_classes",
    }
    torch.save(classifier, cache_dir / "classifier.pt")

    metadata = {
        "dataset": cfg["data"]["dataset"],
        "model": cfg["model"]["name"],
        "feature_layer": cfg["model"].get("feature_layer", "pre_classifier_flattened_pool_output"),
        "checkpoint": "checkpoint_final.pt",
        "cache_files": sorted(path.name for path in cache_dir.glob("*.pt")),
        "split_metadata": data.split_metadata,
    }
    (cache_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

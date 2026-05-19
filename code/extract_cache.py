from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data import build_data_bundle
from models import build_model, get_classifier_state
from train_utils import load_config, set_seed


def _unpack_batch(batch):
    if len(batch) == 3:
        images, labels, indices = batch
        return images, labels, indices
    images, labels = batch
    return images, labels, None


def _expand_relative(path: str | Path, run_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate


def _checkpoint_filename_from_tag(tag: str) -> str:
    if tag.endswith(".pt"):
        return tag
    if tag.startswith("checkpoint_"):
        return f"{tag}.pt"
    return f"checkpoint_{tag}.pt"


def resolve_checkpoint_path(run_dir: Path, checkpoint_arg: str | None, checkpoint_tag: str | None) -> Path:
    if checkpoint_arg:
        path = _expand_relative(checkpoint_arg, run_dir)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    tag = checkpoint_tag or "final"
    candidates = [
        run_dir / "checkpoints" / _checkpoint_filename_from_tag(tag),
        run_dir / _checkpoint_filename_from_tag(tag),
    ]
    if tag == "final":
        candidates.append(run_dir / "checkpoint_final.pt")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not resolve checkpoint tag {tag!r}; tried {[str(path) for path in candidates]}"
    )


def infer_checkpoint_tag(path: Path, checkpoint_tag: str | None) -> str:
    if checkpoint_tag:
        return checkpoint_tag[:-3] if checkpoint_tag.endswith(".pt") else checkpoint_tag
    stem = path.stem
    if stem.startswith("checkpoint_"):
        return stem[len("checkpoint_") :]
    return stem


def resolve_cache_dir(run_dir: Path, cache_dir_arg: str | None, checkpoint_tag: str) -> Path:
    if cache_dir_arg:
        return _expand_relative(cache_dir_arg, run_dir)
    return run_dir / "cache" / checkpoint_tag


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


def _checkpoint_payload(raw: Any, checkpoint_path: Path) -> dict[str, Any]:
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw
    raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}; expected model_state_dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract shared logits/features cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", help="Explicit checkpoint path, absolute or relative to --run-dir")
    parser.add_argument("--checkpoint-tag", help="Checkpoint tag such as final, best_val, or epoch_0050")
    parser.add_argument("--cache-dir", help="Explicit output cache dir; default is <run-dir>/cache/<checkpoint-tag>")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("run", {}).get("seed", 0))
    set_seed(seed)
    run_dir = Path(args.run_dir)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint, args.checkpoint_tag)
    checkpoint_tag = infer_checkpoint_tag(checkpoint_path, args.checkpoint_tag)
    cache_dir = resolve_cache_dir(run_dir, args.cache_dir, checkpoint_tag)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = build_data_bundle(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = _checkpoint_payload(torch.load(checkpoint_path, map_location=device), checkpoint_path)
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

    classifier = get_classifier_state(model)
    torch.save(classifier, cache_dir / "classifier.pt")

    data_cfg = cfg.get("data", {})
    id_dataset = data_cfg.get("id_dataset", data_cfg.get("dataset", data.split_metadata.get("id_dataset")))
    metadata = {
        "dataset": id_dataset,
        "id_dataset": id_dataset,
        "ood_datasets": sorted(data.ood_test_loaders.keys()),
        "model": cfg["model"]["name"],
        "feature_layer": cfg["model"].get("feature_layer", "pre_classifier_flattened_pool_output"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "cache_tag": checkpoint_tag,
        "cache_dir": str(cache_dir),
        "cache_files": sorted(path.name for path in cache_dir.glob("*.pt")),
        "classifier": {
            "orientation": classifier["orientation"],
            "classifier_attr": classifier.get("classifier_attr"),
        },
        "normalization": data.normalization,
        "split_metadata": data.split_metadata,
    }
    (cache_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .registry import DatasetSpec, get_dataset_spec
from .splits import (
    class_counts_for_indices,
    hash_indices,
    make_stratified_train_val_indices,
    make_train_val_indices,
)
from .transforms import eval_transform, train_transform


class DatasetWithIndex(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        return image, int(target), int(idx)


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(map(int, indices))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        source_idx = self.indices[idx]
        image, target = self.dataset[source_idx]
        return image, int(target), int(source_idx)


@dataclass
class DataBundle:
    train_aug_loader: DataLoader
    id_train_fit_loader: DataLoader
    id_val_loader: DataLoader
    id_test_loader: DataLoader
    ood_test_loaders: dict[str, DataLoader]
    num_classes: int
    normalization: dict[str, list[float] | str | bool]
    split_metadata: dict[str, Any]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def _targets(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(x) for x in getattr(dataset, "targets")]
    if hasattr(dataset, "labels"):
        return [int(x) for x in getattr(dataset, "labels")]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in getattr(dataset, "samples")]
    raise ValueError(f"Cannot infer labels for dataset type {type(dataset).__name__}")


def _make_transform(
    spec: DatasetSpec,
    norm_spec: DatasetSpec,
    target_image_size: int,
    train: bool,
    aug_cfg: dict[str, Any],
):
    if train:
        return train_transform(
            norm_spec.mean,
            norm_spec.std,
            image_size=target_image_size,
            source_image_size=spec.image_size,
            random_crop=aug_cfg.get("random_crop", target_image_size),
            padding=int(aug_cfg.get("padding", 4)),
            horizontal_flip=bool(aug_cfg.get("horizontal_flip", True)),
            grayscale_to_rgb=spec.grayscale_to_rgb,
        )
    return eval_transform(
        norm_spec.mean,
        norm_spec.std,
        image_size=target_image_size,
        source_image_size=spec.image_size,
        grayscale_to_rgb=spec.grayscale_to_rgb,
    )


def _build_dataset(
    dataset_name: str,
    root: Path,
    transform,
    train: bool,
    download: bool,
    split: str | None = None,
    imagefolder_root: Path | None = None,
):
    name = dataset_name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=str(root), train=train, transform=transform, download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root=str(root), train=train, transform=transform, download=download)
    if name == "svhn":
        svhn_split = split or ("train" if train else "test")
        return datasets.SVHN(root=str(root), split=svhn_split, transform=transform, download=download)
    if name == "mnist":
        return datasets.MNIST(root=str(root), train=train, transform=transform, download=download)
    if name == "tiny_imagenet":
        if imagefolder_root is None:
            raise ValueError(
                "tiny_imagenet requires an ImageFolder root via ood entry root/path or data.tiny_imagenet_root"
            )
        if not imagefolder_root.exists():
            raise FileNotFoundError(f"TinyImageNet ImageFolder root does not exist: {imagefolder_root}")
        return datasets.ImageFolder(root=str(imagefolder_root), transform=transform)
    raise ValueError(f"Unknown dataset name {dataset_name!r}")


def _normalize_ood_entry(entry: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"name": entry.lower(), "dataset": entry.lower()}
    raw_dataset = entry.get("dataset", entry.get("name"))
    if raw_dataset is None or str(raw_dataset).strip() == "":
        raise ValueError(f"OOD entry must include name or dataset: {entry}")
    dataset_name = str(raw_dataset).lower()
    return {**entry, "dataset": dataset_name, "name": str(entry.get("name", dataset_name)).lower()}


def _imagefolder_root(data_cfg: dict[str, Any], entry: dict[str, Any]) -> Path | None:
    root_value = entry.get("root", entry.get("path"))
    if root_value is None and entry.get("dataset") == "tiny_imagenet":
        root_value = data_cfg.get("tiny_imagenet_root")
    if root_value is None:
        root_value = data_cfg.get("ood_roots", {}).get(str(entry.get("dataset")))
    return _expand_path(root_value) if root_value is not None else None


def build_data_bundle(cfg: dict[str, Any]) -> DataBundle:
    data_cfg = cfg["data"]
    id_dataset_name = str(data_cfg.get("id_dataset", data_cfg.get("dataset", "cifar10"))).lower()
    id_spec = get_dataset_spec(id_dataset_name)
    norm_name = str(data_cfg.get("normalization", "id_dataset")).lower()
    norm_spec = id_spec if norm_name in ("id_dataset", id_dataset_name) else get_dataset_spec(norm_name)

    root = _expand_path(data_cfg.get("root", "${HOME}/datasets"))
    batch_size = int(data_cfg.get("batch_size", 128))
    eval_batch_size = int(data_cfg.get("eval_batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    split_seed = int(data_cfg.get("split_seed", cfg.get("run", {}).get("seed", 0)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    seed = int(cfg.get("run", {}).get("seed", 0))
    download = bool(data_cfg.get("download", True))
    target_image_size = int(data_cfg.get("image_size", 32))
    aug_cfg = data_cfg.get("train_augmentation", {})

    train_aug_dataset = _build_dataset(
        id_dataset_name,
        root,
        _make_transform(id_spec, norm_spec, target_image_size, train=True, aug_cfg=aug_cfg),
        train=True,
        download=download,
        imagefolder_root=_imagefolder_root(data_cfg, {"dataset": id_dataset_name}),
    )
    train_eval_dataset = _build_dataset(
        id_dataset_name,
        root,
        _make_transform(id_spec, norm_spec, target_image_size, train=False, aug_cfg=aug_cfg),
        train=True,
        download=download,
        imagefolder_root=_imagefolder_root(data_cfg, {"dataset": id_dataset_name}),
    )
    test_dataset = _build_dataset(
        id_dataset_name,
        root,
        _make_transform(id_spec, norm_spec, target_image_size, train=False, aug_cfg=aug_cfg),
        train=False,
        download=download,
        split="test",
        imagefolder_root=_imagefolder_root(data_cfg, {"dataset": id_dataset_name}),
    )

    split_cfg = data_cfg.get("split", {})
    split_type = str(split_cfg.get("type", data_cfg.get("split_type", "random"))).lower()
    labels = _targets(train_eval_dataset)
    if split_type == "stratified":
        split = make_stratified_train_val_indices(labels, val_fraction=val_fraction, seed=split_seed)
    elif split_type == "random":
        split = make_train_val_indices(
            num_samples=len(train_aug_dataset), val_fraction=val_fraction, seed=split_seed
        )
    else:
        raise ValueError(f"Unsupported data split type {split_type!r}")

    train_aug = IndexedSubset(train_aug_dataset, split.train)
    id_train_fit = IndexedSubset(train_eval_dataset, split.train)
    id_val = IndexedSubset(train_eval_dataset, split.val)
    id_test = DatasetWithIndex(test_dataset)

    ood_test_loaders: dict[str, DataLoader] = {}
    for raw_entry in data_cfg.get("ood_datasets", []):
        entry = _normalize_ood_entry(raw_entry)
        ood_name = entry["name"]
        dataset_name = entry["dataset"]
        ood_spec = get_dataset_spec(dataset_name)
        split_name = str(entry.get("split", "test"))
        ood_dataset = _build_dataset(
            dataset_name,
            root,
            _make_transform(ood_spec, norm_spec, target_image_size, train=False, aug_cfg=aug_cfg),
            train=split_name == "train",
            download=download,
            split=split_name,
            imagefolder_root=_imagefolder_root(data_cfg, entry),
        )
        ood_test_loaders[ood_name] = _loader(
            DatasetWithIndex(ood_dataset),
            eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

    split_metadata = {
        "dataset": id_dataset_name,
        "id_dataset": id_dataset_name,
        "num_classes": id_spec.num_classes,
        "split_type": split_type,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "num_total_train": len(train_aug_dataset),
        "num_train": len(split.train),
        "num_val": len(split.val),
        "train_indices_hash": hash_indices(split.train),
        "val_indices_hash": hash_indices(split.val),
        "train_class_counts": class_counts_for_indices(labels, split.train),
        "val_class_counts": class_counts_for_indices(labels, split.val),
        "ood_datasets": sorted(ood_test_loaders.keys()),
    }

    return DataBundle(
        train_aug_loader=_loader(train_aug, batch_size, True, num_workers, pin_memory, seed=seed),
        id_train_fit_loader=_loader(id_train_fit, eval_batch_size, False, num_workers, pin_memory, seed=seed),
        id_val_loader=_loader(id_val, eval_batch_size, False, num_workers, pin_memory, seed=seed),
        id_test_loader=_loader(id_test, eval_batch_size, False, num_workers, pin_memory, seed=seed),
        ood_test_loaders=ood_test_loaders,
        num_classes=id_spec.num_classes,
        normalization={
            "name": norm_spec.name,
            "mean": list(norm_spec.mean),
            "std": list(norm_spec.std),
            "ood_uses_id_normalization": True,
        },
        split_metadata=split_metadata,
    )

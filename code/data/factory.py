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

from .splits import hash_indices, make_train_val_indices
from .transforms import CIFAR10_MEAN, CIFAR10_STD, cifar10_eval_transform, cifar10_train_transform


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
    normalization: dict[str, list[float]]
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


def build_data_bundle(cfg: dict[str, Any]) -> DataBundle:
    data_cfg = cfg["data"]
    if data_cfg.get("dataset") != "cifar10":
        raise ValueError("Milestone 1A supports only data.dataset=cifar10")

    root = Path(os.path.expandvars(str(data_cfg.get("root", "${HOME}/datasets")))).expanduser()
    batch_size = int(data_cfg.get("batch_size", 128))
    eval_batch_size = int(data_cfg.get("eval_batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    split_seed = int(data_cfg.get("split_seed", cfg.get("run", {}).get("seed", 0)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    seed = int(cfg.get("run", {}).get("seed", 0))

    train_aug_dataset = datasets.CIFAR10(
        root=str(root), train=True, transform=cifar10_train_transform(), download=True
    )
    train_eval_dataset = datasets.CIFAR10(
        root=str(root), train=True, transform=cifar10_eval_transform(), download=True
    )
    test_dataset = datasets.CIFAR10(
        root=str(root), train=False, transform=cifar10_eval_transform(), download=True
    )

    split = make_train_val_indices(
        num_samples=len(train_aug_dataset), val_fraction=val_fraction, seed=split_seed
    )

    train_aug = IndexedSubset(train_aug_dataset, split.train)
    id_train_fit = IndexedSubset(train_eval_dataset, split.train)
    id_val = IndexedSubset(train_eval_dataset, split.val)
    id_test = DatasetWithIndex(test_dataset)

    ood_test_loaders: dict[str, DataLoader] = {}
    for name in data_cfg.get("ood_datasets", []):
        if name.lower() != "svhn":
            raise ValueError(f"Milestone 1A supports only SVHN OOD, got {name}")
        svhn = datasets.SVHN(
            root=str(root),
            split="test",
            transform=cifar10_eval_transform(),
            download=True,
        )
        ood_test_loaders["svhn"] = _loader(
            DatasetWithIndex(svhn),
            eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

    split_metadata = {
        "dataset": "cifar10",
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "num_total_train": len(train_aug_dataset),
        "num_train": len(split.train),
        "num_val": len(split.val),
        "train_indices_hash": hash_indices(split.train),
        "val_indices_hash": hash_indices(split.val),
        "ood_datasets": sorted(ood_test_loaders.keys()),
    }

    return DataBundle(
        train_aug_loader=_loader(
            train_aug, batch_size, True, num_workers, pin_memory, seed=seed
        ),
        id_train_fit_loader=_loader(
            id_train_fit, eval_batch_size, False, num_workers, pin_memory, seed=seed
        ),
        id_val_loader=_loader(id_val, eval_batch_size, False, num_workers, pin_memory, seed=seed),
        id_test_loader=_loader(
            id_test, eval_batch_size, False, num_workers, pin_memory, seed=seed
        ),
        ood_test_loaders=ood_test_loaders,
        num_classes=10,
        normalization={
            "name": "cifar10",
            "mean": list(CIFAR10_MEAN),
            "std": list(CIFAR10_STD),
            "ood_uses_id_normalization": True,
        },
        split_metadata=split_metadata,
    )

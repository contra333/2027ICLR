from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]


def make_train_val_indices(
    num_samples: int, val_fraction: float, seed: int
) -> SplitIndices:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    num_val = int(round(num_samples * val_fraction))
    val = np.sort(indices[:num_val]).astype(int).tolist()
    train = np.sort(indices[num_val:]).astype(int).tolist()
    return SplitIndices(train=train, val=val)



def make_stratified_train_val_indices(
    labels: Sequence[int], val_fraction: float, seed: int
) -> SplitIndices:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)
    train_parts = []
    val_parts = []
    for cls in np.unique(labels_np):
        cls_indices = np.where(labels_np == cls)[0]
        rng.shuffle(cls_indices)
        num_val = int(round(len(cls_indices) * val_fraction))
        val_parts.append(cls_indices[:num_val])
        train_parts.append(cls_indices[num_val:])
    train = np.sort(np.concatenate(train_parts)).astype(int).tolist()
    val = np.sort(np.concatenate(val_parts)).astype(int).tolist()
    return SplitIndices(train=train, val=val)


def class_counts_for_indices(labels: Sequence[int], indices: Sequence[int]) -> dict[str, int]:
    labels_np = np.asarray(labels)
    out: dict[str, int] = {}
    for cls in np.unique(labels_np):
        out[str(int(cls))] = int(np.sum(labels_np[list(indices)] == cls))
    return out

def hash_indices(indices: Sequence[int]) -> str:
    payload = json.dumps(list(map(int, indices)), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

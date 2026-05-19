from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    image_size: int = 32
    grayscale_to_rgb: bool = False


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)


DATASET_REGISTRY = {
    "cifar10": DatasetSpec("cifar10", 10, CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": DatasetSpec("cifar100", 100, CIFAR100_MEAN, CIFAR100_STD),
    "svhn": DatasetSpec("svhn", 10, SVHN_MEAN, SVHN_STD),
    "mnist": DatasetSpec("mnist", 10, MNIST_MEAN, MNIST_STD, grayscale_to_rgb=True),
    "tiny_imagenet": DatasetSpec("tiny_imagenet", 200, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD, image_size=64),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset name {name!r}. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key]

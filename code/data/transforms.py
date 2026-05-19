from __future__ import annotations

from torchvision import transforms

from .registry import CIFAR10_MEAN, CIFAR10_STD


def train_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    image_size: int = 32,
    source_image_size: int = 32,
    random_crop: int | None = 32,
    padding: int = 4,
    horizontal_flip: bool = True,
    grayscale_to_rgb: bool = False,
):
    steps = []
    if grayscale_to_rgb:
        steps.append(transforms.Grayscale(num_output_channels=3))
    if source_image_size != image_size:
        steps.append(transforms.Resize(image_size))
    if random_crop:
        steps.append(transforms.RandomCrop(int(random_crop), padding=int(padding)))
    if horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def eval_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    image_size: int = 32,
    source_image_size: int = 32,
    grayscale_to_rgb: bool = False,
):
    steps = []
    if grayscale_to_rgb:
        steps.append(transforms.Grayscale(num_output_channels=3))
    if source_image_size != image_size:
        steps.append(transforms.Resize(image_size))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def cifar10_train_transform():
    return train_transform(CIFAR10_MEAN, CIFAR10_STD)


def cifar10_eval_transform():
    return eval_transform(CIFAR10_MEAN, CIFAR10_STD)

"""
Visualization utilities for property-driven machine learning.

This module provides utilities for saving and visualizing training artifacts
such as input images, adversarial examples, and random samples.
"""

import os
import torch
from torchvision.utils import save_image
from typing import Union, Tuple
from ..training.epoch_info import EpochInfoTrain, EpochInfoTest


def denormalize_image(
    x: torch.Tensor,
    mean: Union[torch.Tensor, Tuple[float, ...]],
    std: Union[torch.Tensor, Tuple[float, ...]],
) -> torch.Tensor:
    """Denormalize a tensor image using mean and std.

    Args:
        x: Normalized image tensor.
        mean: Normalization mean values.
        std: Normalization standard deviation values.

    Returns:
        Denormalized image tensor.
    """
    return x * torch.as_tensor(std, device=x.device).view(-1, 1, 1) + torch.as_tensor(
        mean, device=x.device
    ).view(-1, 1, 1)


def save_epoch_images(
    info: Union[EpochInfoTrain, EpochInfoTest],
    epoch: int,
    save_dir: str,
    mean: Union[torch.Tensor, Tuple[float, ...]],
    std: Union[torch.Tensor, Tuple[float, ...]],
) -> None:
    """Save training/test images for visualization.

    Args:
        info: Epoch information containing images to save.
        epoch: Current epoch number.
        save_dir: Directory to save images.
        mean: Normalization mean values for denormalization.
        std: Normalization standard deviation values for denormalization.
    """
    os.makedirs(save_dir, exist_ok=True)

    def save_img(img: torch.Tensor, name: str):
        """Save a single image with denormalization."""
        save_image(denormalize_image(img, mean, std), os.path.join(save_dir, name))

    if isinstance(info, EpochInfoTrain):
        prefix = "train"
    else:
        prefix = "test"

    save_img(info.input_img, f"{epoch}-{prefix}_input.png")
    save_img(info.adv_img, f"{epoch}-{prefix}_adv.png")
    save_img(info.random_img, f"{epoch}-{prefix}_random.png")

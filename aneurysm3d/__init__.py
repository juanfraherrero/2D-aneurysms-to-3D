"""Reconstruct a 3D aneurysm point cloud from 4 grayscale 2D views."""

import torch

from .model import Images2PointCloud


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

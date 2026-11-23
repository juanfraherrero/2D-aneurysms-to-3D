import torch
from torch import nn

NUM_VIEWS = 4
NUM_POINTS = 1500


class Images2PointCloud(nn.Module):
    """Encodes 4 grayscale views (stacked as channels) and decodes a 1500-point cloud in [-1, 1]."""

    def __init__(self, image_size: int = 256):
        super().__init__()
        # 4 stride-2 convs + 1 max-pool => spatial size shrinks by 32
        feat = image_size // 32
        self.encoder = nn.Sequential(
            nn.Conv2d(NUM_VIEWS, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 512, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * feat * feat, 4096), nn.ReLU(),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        self.point_cloud_decoder = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, NUM_POINTS * 3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) -> (B, 1500, 3)"""
        return self.point_cloud_decoder(self.encoder(x)).view(-1, NUM_POINTS, 3)

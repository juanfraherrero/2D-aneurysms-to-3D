from pathlib import Path

import pywavefront
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .model import NUM_VIEWS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_views(image_paths: list[Path], image_size: int) -> torch.Tensor:
    """Load the 4 views as grayscale and stack them as channels: (4, H, W) in [0, 1]."""
    to_tensor = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return torch.cat([to_tensor(Image.open(p).convert("L")) for p in image_paths], dim=0)


def load_obj_points(obj_path: Path) -> torch.Tensor:
    """Load the vertices of an .obj, centered at the origin and scaled into the unit sphere: (N, 3)."""
    scene = pywavefront.Wavefront(str(obj_path), collect_faces=False, parse=True)
    points = torch.tensor([v[:3] for v in scene.vertices], dtype=torch.float32)
    points -= points.mean(dim=0)
    return points / points.norm(dim=1).max()


class AneurysmDataset(Dataset):
    """
    Each sample is a sub-folder of `root` with `images/` (4 views) and `object/` (1 .obj).
    Returns (views (4, H, W), points (N, 3)); N varies per sample.
    """

    def __init__(self, root: str, image_size: int = 256):
        self.image_size = image_size
        self.samples = []
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            images = list_images(folder / "images")
            objects = sorted((folder / "object").glob("*.obj"))
            if len(images) == NUM_VIEWS and len(objects) == 1:
                self.samples.append((images, objects[0]))
            else:
                print(f"Skipping {folder}: expected {NUM_VIEWS} images and 1 .obj")
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, obj = self.samples[idx]
        return load_views(images, self.image_size), load_obj_points(obj)


def collate(batch):
    """Stack images; keep point clouds as a list since they have different sizes."""
    images, points = zip(*batch)
    return torch.stack(images), list(points)

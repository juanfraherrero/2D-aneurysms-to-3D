"""Reconstruct a 3D point cloud from 4 views of an aneurysm.

    python predict.py --images path/to/images --checkpoint runs/0/checkpoint.pt --out aneurysm.ply
"""
import argparse

import torch

from aneurysm3d import Images2PointCloud, get_device
from aneurysm3d.data import list_images, load_views
from aneurysm3d.model import NUM_VIEWS


def save_ply(points, path):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with open(path, "w") as f:
        f.write(header + "\n".join(f"{x} {y} {z}" for x, y, z in points.tolist()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help=f"folder with exactly {NUM_VIEWS} images, sorted by name")
    parser.add_argument("--checkpoint", required=True, help="checkpoint.pt produced by train.py")
    parser.add_argument("--out", default="prediction.ply", help="output .ply point cloud")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    images = list_images(args.images)
    if len(images) != NUM_VIEWS:
        parser.error(f"expected {NUM_VIEWS} images in {args.images}, found {len(images)}")

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = Images2PointCloud(ckpt["image_size"]).to(device).eval()
    model.load_state_dict(ckpt["model"])

    with torch.no_grad():
        points = model(load_views(images, ckpt["image_size"]).unsqueeze(0).to(device))[0]

    save_ply(points.cpu(), args.out)
    print(f"Saved {len(points)} points to {args.out}")


if __name__ == "__main__":
    main()

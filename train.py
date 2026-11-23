"""Train the 2D-views -> 3D point cloud model.

    python train.py                                # uses configs/train.yaml
    python train.py --epochs 5 --data_fraction 0.1 # any config key can be overridden
"""
import argparse
from pathlib import Path

import matplotlib
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from aneurysm3d import Images2PointCloud, get_device
from aneurysm3d.data import AneurysmDataset, collate
from aneurysm3d.loss import batch_chamfer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_config():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/train.yaml")
    args, _ = pre.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre], formatter_class=argparse.RawDescriptionHelpFormatter)
    config = yaml.safe_load(Path(args.config).read_text())
    # every key in the yaml becomes an overridable --flag
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    return parser.parse_args()


def new_run_dir(output: str) -> Path:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    run = root / str(sum(1 for p in root.iterdir() if p.is_dir()))
    run.mkdir()
    return run


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`; trains when an optimizer is given. Returns the mean loss."""
    model.train(optimizer is not None)
    total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            loss = batch_chamfer(model(images.to(device)), [t.to(device) for t in targets])
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / max(len(loader), 1)


def save_plot(history, path):
    plt.figure(figsize=(10, 5))
    for name, values in history.items():
        if values:
            plt.plot(values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Chamfer distance")
    plt.legend()
    plt.savefig(path)
    plt.close()


def main():
    cfg = parse_config()
    device = get_device(cfg.device)

    dataset = AneurysmDataset(cfg.data, cfg.image_size)
    if cfg.data_fraction < 1:
        dataset, _ = random_split(dataset, [cfg.data_fraction, 1 - cfg.data_fraction])
    train_set, val_set = random_split(dataset, [1 - cfg.val_split, cfg.val_split])
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, cfg.batch_size, collate_fn=collate)

    model = Images2PointCloud(cfg.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = {"train": [], "val": []}
    start_epoch = 0

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history, start_epoch = ckpt["history"], ckpt["epoch"] + 1

    run_dir = new_run_dir(cfg.output)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(vars(cfg)))
    print(f"Device: {device} | train: {len(train_set)} | val: {len(val_set)} | saving to {run_dir}")

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Epochs", initial=start_epoch, total=cfg.epochs):
        history["train"].append(run_epoch(model, train_loader, device, optimizer))
        if len(val_set):
            history["val"].append(run_epoch(model, val_loader, device))
        tqdm.write(f"Epoch {epoch + 1}: " + ", ".join(f"{k} {v[-1]:.4f}" for k, v in history.items() if v))

        if (epoch + 1) % cfg.save_every == 0 or epoch + 1 == cfg.epochs:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "history": history, "image_size": cfg.image_size},
                       run_dir / "checkpoint.pt")
            save_plot(history, run_dir / "loss.png")

    print(f"Done. Checkpoint: {run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()

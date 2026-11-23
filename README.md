# 2D Aneurysms → 3D

**Reconstruct the 3D shape of an intracranial aneurysm from four 2D images.**

Given 4 grayscale views of an aneurysm (taken from different camera angles), a small convolutional network predicts a **3D point cloud of 1,500 points** describing its shape. The network is trained by comparing its prediction with the real 3D mesh using the [Chamfer distance](https://en.wikipedia.org/wiki/Chamfer_distance).

```
 4 views (4 × 256 × 256)          CNN encoder         MLP decoder          3D point cloud
 ┌──┐┌──┐┌──┐┌──┐          ──►   ┌──────────┐  ──►  ┌──────────┐  ──►   1500 × (x, y, z)
 └──┘└──┘└──┘└──┘                └──────────┘       └──────────┘         saved as .ply
```

This was developed as part of a university thesis. The 3D models used for training come from the [IntrA dataset](https://github.com/intra3d2019/IntrA).

## Quick start

```bash
git clone --recurse-submodules https://github.com/juanfraherrero/2D-aneurysms-to-3D.git
cd 2D-aneurysms-to-3D
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train (uses GPU automatically if available: CUDA or Apple MPS):

```bash
python train.py --data data/training_data
```

Reconstruct an aneurysm from 4 images:

```bash
python predict.py --images path/to/4_images --checkpoint runs/0/checkpoint.pt --out aneurysm.ply
```

Open the `.ply` in [MeshLab](https://www.meshlab.net/), [ParaView](https://www.paraview.org/) or Blender to view it.

## 1. Prepare the data

Each training sample is a folder with the 4 views and the ground-truth `.obj`:

```
data/training_data/
├── sample_001/
│   ├── images/        # exactly 4 images, read in name order (1.jpg, 2.jpg, ...)
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   └── 4.jpg
│   └── object/
│       └── 1.obj      # the 3D mesh (only its vertices are used)
├── sample_002/
│   └── ...
```

Folders that don't match this layout are skipped with a warning.

### Generating samples from 3D meshes

If you have `.obj` meshes but no images, the [`object-to-images`](https://github.com/juanfraherrero/object-to-images) submodule (in `external/get-images-from-mesh-obj`) renders them from multiple angles with ParaView, producing the layout above. Edit its `config_preparing_data.json` (input/output folders, image size, number of rotations, camera distance…) and run it with ParaView's Python:

```bash
pvpython external/get-images-from-mesh-obj/prepare_data.py -c external/get-images-from-mesh-obj/config_preparing_data.json
```

## 2. Train

All settings live in [`configs/train.yaml`](configs/train.yaml). Any of them can be overridden from the command line:

```bash
python train.py                                   # defaults from configs/train.yaml
python train.py --epochs 5 --data_fraction 0.05   # quick smoke test on 5% of the data
python train.py --resume runs/0/checkpoint.pt     # continue a previous run
python train.py --config my_config.yaml           # use another config file
```

| Setting         | Default              | Description                                        |
| --------------- | -------------------- | -------------------------------------------------- |
| `data`          | `data/training_data` | Folder with the training samples                   |
| `output`        | `runs`               | Each run is saved in `runs/<n>/`                   |
| `image_size`    | `256`                | Images are resized to this (multiple of 32)        |
| `batch_size`    | `8`                  |                                                    |
| `epochs`        | `100`                |                                                    |
| `lr`            | `0.001`              | Adam learning rate                                 |
| `val_split`     | `0.2`                | Fraction of samples used for validation            |
| `data_fraction` | `1.0`                | Use only part of the dataset (for quick tests)     |
| `save_every`    | `5`                  | Save a checkpoint every N epochs                   |
| `device`        | `auto`               | `auto`, `cuda`, `mps` or `cpu`                     |
| `resume`        | `null`               | Checkpoint to continue training from               |

Each run writes to `runs/<n>/`:

- `checkpoint.pt` – model weights, optimizer state and loss history
- `loss.png` – training/validation loss curves
- `config.yaml` – the exact settings used

## 3. Predict

```bash
python predict.py --images my_aneurysm/images --checkpoint runs/0/checkpoint.pt --out my_aneurysm.ply
```

The folder must contain exactly 4 images, in the same order/angles used for training. The output point cloud is centered at the origin and scaled to fit in a unit sphere.

## How it works

| Step     | Details                                                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Input    | 4 grayscale views stacked as channels → tensor `4 × 256 × 256`                                                               |
| Target   | Vertices of the `.obj` mesh, centered and scaled into `[-1, 1]`                                                              |
| Encoder  | 4 strided convolutions + max-pool → fully connected layers → 1024-d latent vector                                            |
| Decoder  | Fully connected layers → `1500 × 3` coordinates, `tanh` keeps them in `[-1, 1]`                                              |
| Loss     | Chamfer distance (sum of nearest-neighbour distances in both directions), so prediction and target can have different sizes |

## Project structure

```
├── train.py              # training entry point
├── predict.py            # inference: 4 images → .ply
├── configs/train.yaml    # default training settings
├── aneurysm3d/
│   ├── model.py          # Images2PointCloud network
│   ├── data.py           # dataset, image and .obj loading
│   └── loss.py           # Chamfer distance
└── external/
    └── get-images-from-mesh-obj/   # submodule: render .obj meshes into training images
```

## Requirements

Python 3.9+ and PyTorch 2.x. A GPU is recommended for training but not required.

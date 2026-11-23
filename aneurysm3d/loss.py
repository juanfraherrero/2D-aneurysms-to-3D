import torch


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between point clouds a (N, 3) and b (M, 3)."""
    dist = torch.cdist(a, b)
    return dist.min(dim=1).values.sum() + dist.min(dim=0).values.sum()


def batch_chamfer(pred: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
    """Mean Chamfer distance over a batch where each target may have a different number of points."""
    return torch.stack([chamfer_distance(p, t) for p, t in zip(pred, targets)]).mean()

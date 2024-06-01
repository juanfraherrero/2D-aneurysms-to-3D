import torch

class ChamferDistancePy(torch.nn.Module):
  def __init__(self):
      super(ChamferDistancePy, self).__init__()

  def forward (self, points1, points2):
    """
    Calculate Chamfer Distance between two point clouds.

    :param points1: Tensor of shape (N, 3)
    :param points2: Tensor of shape (M, 3)
    :return: Chamfer Distance
    """
    # Compute pairwise distance matrix
    dist_matrix = torch.cdist(points1, points2, p=2)  # shape (N, M)

    return torch.sum(torch.min(dist_matrix, dim=1)[0]) + torch.sum(torch.min(dist_matrix, dim=0)[0])
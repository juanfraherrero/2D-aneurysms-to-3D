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

    # For each point in points1, find the nearest point in points2
    min_dist1, _ = torch.min(dist_matrix, dim=1)
    # For each point in points2, find the nearest point in points1
    min_dist2, _ = torch.min(dist_matrix, dim=0)

    # Sum of minimum distances
    chamfer_dist = torch.sum(min_dist1) + torch.sum(min_dist2)

    return chamfer_dist
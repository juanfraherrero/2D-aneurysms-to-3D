import pywavefront
import numpy as np
import torch

class escaleObject(torch.nn.Module):
  """Transformador que que escala una matriz de tensores entre -1 y 1"""
  
  def __init__(self):
    super(escaleObject, self).__init__()

  def forward(self, matrixInGpu):     
    # calculate the max distance for the points from origin
    # use torch.linalg.vector_norm() use euclidean norm for each row or point
    radii = torch.linalg.vector_norm(matrixInGpu, dim=1)
    
    # find the max of these norms
    max_radius = torch.max(radii)

    # divide the matrix by the max_radius
    normalized_points = matrixInGpu / max_radius

    return normalized_points

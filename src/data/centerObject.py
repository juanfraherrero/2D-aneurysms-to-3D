import pywavefront
import numpy as np
import torch

class centerObject(torch.nn.Module):
  """Transformador que centraliza una matriz de tensores."""
  
  def __init__(self, useCuda=False):
    super(centerObject, self).__init__()
    self.useCuda = useCuda

  def forward(self, matrixInGpu):     
    # Convertir a Tensor de PyTorch
    # Paso 1: Calcular el centroide
    centroid = torch.mean(matrixInGpu, dim=0)  # Promedio a lo largo de la dimensión 0 (n puntos)
    # Paso 2: Restar el centroide de cada punto
    centered_points = matrixInGpu - centroid

    return centered_points

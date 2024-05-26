import pywavefront
import numpy as np
import torch

class tensorToDevice(torch.nn.Module):
  """Transformador que carga un archivo .obj y lo convierte a tensor."""
  
  def __init__(self, useCuda=False):
    super(tensorToDevice, self).__init__()
    self.useCuda = useCuda

  def forward(self, tensorInCpu):     
    # Convertir a Tensor de PyTorch
    if(self.useCuda):
      tensorInCpu = tensorInCpu.to('cuda')
    return tensorInCpu

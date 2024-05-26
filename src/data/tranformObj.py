import pywavefront
import numpy as np
import torch

class LoadOBJTransform(torch.nn.Module):
  """Transformador que carga un archivo .obj y lo convierte a tensor."""
  
  def __init__(self, ):
    super(LoadOBJTransform, self).__init__()

  def forward(self, file_path):
    # Cargar el modelo 3D
    scene = pywavefront.Wavefront(file_path, collect_faces=False)

    vertices = []
    # Cargar los vértices de los materiales
    for name, material in scene.materials.items():
        for i in range(0, len(material.vertices), 3):
            vertices.append([material.vertices[i], material.vertices[i+1], material.vertices[i+2]])
        
    # Convertir a Tensor de PyTorch
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    return vertices_tensor

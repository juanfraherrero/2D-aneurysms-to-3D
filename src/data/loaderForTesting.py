from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import math
from data.tranformObj import LoadOBJTransform
from data.tensorToDevice import tensorToDevice
from data.centerObject import centerObject
from data.escaleObject import escaleObject

# Dataset Personalizado para cargar imágenes
class ImagesDatasetForTesting(Dataset):
    def __init__(self, folder_path, size=(512, 512), useCuda=False):
        """
        Inicializa el dataset con las imágenes de la ruta.
        
        Args:
            folder_path (str): Ruta al directorio que contiene las imágenes.
            size (tuple): Dimensiones a las que se redimensionarán las imágenes. (512, 512) por defecto
            useCuda (bool): Si se utilizará CUDA para el entrenamiento. False por defecto.
        """
        self.useCuda = useCuda
        self.size = size
        full_path = os.path.join(os.getcwd(), folder_path)
        # load images
        self.files = [os.path.join(full_path,path) for path in os.listdir(full_path) if os.path.isfile(os.path.join(full_path,path))]
        
        if not self.files:
            raise RuntimeError(f"No se encontraron imágenes en {folder_path}")

        if len(self.files) != 4:
            raise RuntimeError(f"El directorio {folder_path} debe tener solo 4 imagenes")
        
        # set the transformation to apply to the images
        self.transform_image = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            tensorToDevice(self.useCuda)
        ])

    def __len__(self):
        return 1 # solo se tiene un elemento

    def __getitem__(self, idx):
        """
        Obtiene las 4 imágenes agrupadas transformandolas, concatenando por canáles y devuelve.
        
        Returns:
            tensor: Imágenes transformada en forma de tensor.
        """

        images = [self.transform_image(Image.open(path).convert('RGB')) for path in self.files] # transformamos las imágenes
        combined_images = torch.cat(images, dim=0) # concatenamos las imágenes por canales
        return combined_images
        

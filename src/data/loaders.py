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
class ImagesDataset(Dataset):
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
        full_path = folder_path
        # load all folders of each dataset
        self.dataset = [os.path.join(full_path,path) for path in os.listdir(full_path) if os.path.isdir(os.path.join(full_path,path))]
        self.files = []
        for folder in self.dataset:
            image_full_path = os.path.join(folder, "images")
            object_full_path = os.path.join(folder, "object")
            images = [os.path.join(image_full_path,x) for x in sorted(os.listdir(image_full_path))]
            object = [os.path.join(object_full_path,x) for x in sorted(os.listdir(object_full_path))]
            if len(images) == 4 and len(object) == 1:
                self.files.append([images, object[0]]) # add the images as an array and the object to the list
            else:
                print(f"La carpeta {folder} no tiene las imágenes ni objeto necesarios")
                break
        
        if not self.files:
            raise RuntimeError(f"No se encontraron imágenes en {folder_path}")

        self.datasetLenght = len(self.files)  # número de imágenes en el dataset
        # set the transformation to apply to the images
        self.transform_image = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            tensorToDevice(self.useCuda)
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) do not utilize this for now, with ToTensor() already normalize the image to 0~1
        ])
        self.retrieve_tranform_object = transforms.Compose([
            LoadOBJTransform(),
            tensorToDevice(self.useCuda),
            centerObject(),
            escaleObject()
        ])

    def __len__(self):
        return self.datasetLenght

    def __getitem__(self, idx):
        """
        Obtiene 4 imágenes dado un índice, las transforma, concatena por canáles y devuelve.
        
        Args:
            idx (int): Índice de la imagen a cargar.
        
        Returns:
            tensor: Imágenes transformada en forma de tensor.
        """
        data_array = self.files[idx]  # obtenemos la idx imágenes y objecto
        image_paths = data_array[0]
        object_path = data_array[1]
        images = [self.transform_image(Image.open(path).convert('L')) for path in image_paths] # transformamos las imágenes
        combined_images = torch.cat(images, dim=0) # concatenamos las imágenes por canales
        object_tranform = self.retrieve_tranform_object(object_path)
        return combined_images, object_tranform 
        

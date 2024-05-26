import torch
from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch):
    """ Collate function to handle batches of images and variable-size point sets. """
    
    # Separa los datos en dos listas, una para las imágenes y otra para las nubes de puntos
    images = [item[0] for item in batch]  # Listar todas las imágenes
    point_sets = [item[1] for item in batch]  # Listar todas las nubes de puntos
    
    # junta en un tensor las imágenes del batch, las mantiene en el mismo device que estén
    images_stacked = torch.stack(images)
    
    # Las nubes de puntos se manejan como una lista de tensores, ya que varían en tamaño
        
    return images_stacked, point_sets


import torch
from torch import nn

class CombinedImagesAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # input is 4 imágenes de 1(b&w)-- 12 canales x 256(height) x 256(witdh), 
            
            nn.Conv2d(4, 32, 3, stride=2, padding=1),  # output is 32 x 128 x 128
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # output is 64 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output is 64 x 32 x 32
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # output is 128 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 512, 3, stride=2, padding=1), # output is 512 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(512, 2048, 3, stride=2, padding=1), # output is 2048 x 4 x 4
            nn.ReLU(),
            # batch normalization could be applied here
            nn.Flatten(),
            nn.Linear(2048*4*4, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 256),  # This will be the encoded representation
        )
        
        # Decoder para nube de puntos
        self.point_cloud_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(2048, 4500),  # Produce 3 valores (x, y, z) para cada uno de los puntos 1500 puntos
            nn.Tanh()  # Normaliza los valores entre -1 y 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.point_cloud_decoder(x)
        # cambiar la forma de x para que sea de 1024 puntos con 3 valores cada uno
        x = x.view(-1, 1500, 3)
        return x
    
    def concatImages(self, arrOfImages):
        """
        Dado un arreglo de tensores los concatenará en un solo tensor.
        Los concatenará en la dimensión 0. que es la dimensión de los canales.

        Args:
            arrOfImages (array<Tensor>): Arreglo de tensores

        Returns:
            torch.Tensor: Tensor con las imágenes concatenadas.
        """
        return torch.cat(arrOfImages, 0)

    # OVERRIDE STRING METHOD TO ADD INPUT SIZE
    def __str__(self):
        original_str = super(CombinedImagesAutoencoder, self).__str__()  # call original str mnethod
        custom_str = 'Input is 4 images with 1 channel (L) --> 4 channels x 256(height) x 256(witdh). \n\n'
        return custom_str + original_str  # Combina tu información personalizada con la representación estándar

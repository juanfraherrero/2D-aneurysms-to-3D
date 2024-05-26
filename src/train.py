import os

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from modules.Model import CombinedImagesAutoencoder

from data.loaders import ImagesDataset
from data.customCollate import custom_collate_fn

from external.chamferDist import ChamferDistance

from utils import utils

import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# Function to train the model
def train(model, train_loader, eval_loader, epochs, charts_path, learning_rate):
    '''
        Use Adam optimizer with 0.0001 learning rate
        Use ChamferDistance as loss function in bidireccional mode

    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lossFunction = ChamferDistance()
    
    # metrics to plot
    train_losses = []
    valid_losses = []

    progress_bar_epochs = tqdm(range(epochs), desc='Processing data in epochs', unit="Epoch", leave=False)

    model.train()
    for epoch in range(epochs):        
        # bucle to train the model
        progress_bar_train = tqdm(total=len(train_loader), desc='Processing training batches', unit='batch', leave=True)
        train_loss = 0
        for data, target in train_loader:

            optimizer.zero_grad() # Clean gradients

            # get predictions
            output = model(data)

            # calculate the batch loss
            batch_loss = 0
            # cicle over the batch size
            for idx in range(len(target)):
                # get target, ouput and add one dimension --- here can be parallelized
                gt = target[idx].unsqueeze(0) 
                out = output[idx].unsqueeze(0)
                batch_loss += lossFunction(out, gt, bidirectional=True)
            
            # average loss in batch
            batch_loss /= len(target) 


            batch_loss.backward() # Calculate gradients
            optimizer.step() # Update weights
            train_loss += batch_loss.item() # Add the loss to the total loss
            progress_bar_train.update(1)
        
        # close the progress bar 
        progress_bar_train.close()
        
        # average loss per epoch
        train_losses.append(train_loss / len(train_loader))



        # evaluate the model in each epoch if evaluation is setted
        if(eval_loader is not None):
            model.eval()
            with torch.no_grad(): # avoid change gradients
                
                progress_bar_eval = tqdm(total=len(eval_loader), desc='Processing eval batches', unit='batch', leave=True)
                valid_loss = 0
                # bucle to eval the model
                for data, target in eval_loader:
                    
                    # get predictions
                    output = model(data)

                    batch_loss = 0
                    # cicle over the batch size
                    for idx in range(len(target)):
                        # get target, ouput and add one dimension --- here can be parallelized
                        gt = target[idx].unsqueeze(0) 
                        out = output[idx].unsqueeze(0)
                        batch_loss += lossFunction(out, gt, bidirectional=True)
                    
                    # average loss in batch 
                    batch_loss /= len(target) 
                    
                    valid_loss += batch_loss.item() # Add the loss to the total loss

                    progress_bar_eval.update(1)
                
                # close the progress bar
                progress_bar_eval.close()

                # average loss per epoch
                valid_losses.append(valid_loss / len(eval_loader))

            print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Valid Loss: {valid_losses[-1]}")
        else:
            print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}")
        
        # update the progress bar
        progress_bar_epochs.update(1)
    
    # end the progress bar
    progress_bar_epochs.close()

    if charts_path is None:
        return
    
    # plot the losses and save it in dir
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(charts_path,'train_valid_losses.png'))
                

# Código principal
if __name__ == '__main__':
    # load config yaml
    with open('src/config_train.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    if config["useCuda"] == True:
        useCuda = torch.cuda.is_available()
    else:
        useCuda = False
    print (useCuda)
    # Asumiendo que las imágenes están en 'path_to_images'
    dataset = ImagesDataset(folder_path=config["folder_data_path"], size=config["image_size"], useCuda=useCuda)
    # Definir tamaños para entrenamiento y validación
    total_size = len(dataset)
    train_size = int(config["train_percentage_split"] * total_size)
    valid_size = total_size - train_size
    train_dataset, eval_dataset, trash_dataset = random_split(dataset, [0.9, 0.1,0.0]) # verify if no data losses
    
    print(f"Total size: {total_size} Train size: {len(train_dataset)}, Validation size: {len(eval_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=custom_collate_fn)
    
    model = CombinedImagesAutoencoder()
    if(useCuda):
        model = model.to('cuda')

    # Get folder to save info of training
    folder_path, models_path, results_pat, charts_path = utils.get_folder_for_training_model()

    train(model, train_loader, eval_loader, epochs=config["epochs"], charts_path=charts_path, learning_rate=config["learning_rate"])

    # Guardar el modelo
    torch.save(model.state_dict(), os.path.join(models_path,'model.pth'))
    
    # Save model in txt 
    model_info_path = os.path.join(models_path,'model_info.txt')
    with open(model_info_path, 'w') as file:
        file.write(str(model))

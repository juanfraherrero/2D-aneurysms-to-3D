# from external.chamferDist import ChamferDistance
import os
import sys
import torch
from torch import optim
from external.chamferDistPython.chamferDistPy import ChamferDistancePy
from utils.save_model import saveModel
from utils.generate_chart import generateChart

# Function to train the model
def train(model, train_loader, eval_loader, epochs, models_path,charts_path, learning_rate, isRunningInColab):
    '''
        Train the model with the train_loader and evaluate with eval_loader
        Use ChamferDistance as loss function
    '''
    # define tqdm by the environment
    if(isRunningInColab):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # check if have to evaluate
    haveToEvaluate = eval_loader is not None

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # lossFunctio
    lossFunction = ChamferDistancePy()

    # metrics to plot
    train_losses = []
    valid_losses = []

    progress_bar_epochs = tqdm(range(epochs), desc='Processing data in epochs', unit="Epoch", leave=True)

    # TRAINING
    model.train()
    for epoch in range(epochs):        
        # bucle to train the model
        progress_bar_train = tqdm(total=len(train_loader), desc='Processing training batches', unit='batch', leave=False)
        train_loss = 0
        for data, target in train_loader:

            # Clean gradients
            optimizer.zero_grad() 

            # Get predictions
            output = model(data)

            # Cicle over the batch size and calculate loss
            batch_loss = 0

            for idx in range(len(target)):
                # Add one dimension to target and ouput --- here can be parallelized¿?
                tg = target[idx].unsqueeze(0) 
                out = output[idx].unsqueeze(0)
                batch_loss += lossFunction(tg, out)
            
            # Average loss in batch
            batch_loss /= len(target) 


            # Calculate gradients
            batch_loss.backward() 
            
            # Update weights
            optimizer.step() 
            
            train_loss += batch_loss.item() # Add the loss to the total loss
            progress_bar_train.update(1)
        
        # close the progress bar 
        progress_bar_train.close()
        
        # average loss per epoch
        train_losses.append(train_loss / len(train_loader))



        # evaluate the model in each epoch if evaluation is setted
        if(haveToEvaluate):
            model.eval()
            # avoid change gradients
            with torch.no_grad():     
                progress_bar_eval = tqdm(total=len(eval_loader), desc='Processing eval batches', unit='batch', leave=False)
                
                valid_loss = 0
                
                # Cicles for each batch in eval
                for data, target in eval_loader:
                    
                    # get predictions
                    output = model(data)

                    batch_loss = 0
                    # cicle over the batch size
                    for idx in range(len(target)):
                        # add one dimension to target and ouput  --- here can be parallelized
                        tg = target[idx].unsqueeze(0) 
                        out = output[idx].unsqueeze(0)
                        batch_loss += lossFunction(tg, out)
                    
                    # average loss in batch 
                    batch_loss /= len(target) 
                    
                    valid_loss += batch_loss.item() # Add the loss to the total loss

                    progress_bar_eval.update(1)
                
                # close the progress bar
                progress_bar_eval.close()

                # average loss per epoch
                valid_losses.append(valid_loss / len(eval_loader))

            tqdm.write(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Valid Loss: {valid_losses[-1]}")
        else:
            tqdm.write(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}")
        
        # update the progress bar
        progress_bar_epochs.update(1)

        saveModel(model, models_path, epoch)
        generateChart(charts_path, train_losses, valid_losses, epoch)
    
    # end the progress bar
    progress_bar_epochs.close()
     
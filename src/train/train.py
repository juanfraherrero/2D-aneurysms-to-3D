# from external.chamferDist import ChamferDistance
import os
import sys
import torch
from torch import optim
from external.chamferDistPython.chamferDistPy import ChamferDistancePy
from utils.save_model import saveModel
from utils.generate_chart import generateChart
from utils.loadCheckpoint import loadCheckpoint 
from utils.zeroGradients import zeroGradients
from utils.getPrediction import getPrediction
from utils.calculateLoss import calculateLoss
from utils.calculate_backpropagation_step import calculateBackpropagationStep
import torch.autograd.profiler as profiler

# Function to train the model
def train(model, train_loader, eval_loader, epochs, models_path,charts_path, learning_rate, isRunningInColab, resume_training, checkpoint_path, useCuda):
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

    # lossFunction
    lossFunction = ChamferDistancePy()

    # metrics to plot
    train_losses = []
    valid_losses = []

    start_epoch = 0
    
    # check if have to resume training
    if(resume_training):
        model, optimizer_checkpoint, start_epoch, train_losses, valid_losses  = loadCheckpoint(model, checkpoint_path)
        print(f"Resume training from epoch {start_epoch} to epoch {epochs}")

    # move the model to cuda if available
    if(useCuda):
        model = model.to('cuda')
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if(resume_training):
        # if resume training, load the optimizer state
        optimizer.load_state_dict(optimizer_checkpoint)

    progress_bar_epochs = tqdm(range(start_epoch, epochs), desc='Processing data in epochs', unit="Epoch", leave=True, initial=start_epoch)

    # TRAINING
    model.train()
    for epoch in range(start_epoch, epochs):        
        # bucle to train the model
        # progress_bar_train = tqdm(total=len(train_loader), desc='Processing training batches', unit='batch', leave=False)
        train_loss = 0
        for data, target in train_loader:

            # Clean gradients
            zeroGradients(optimizer)

            # Get predictions
            output = getPrediction(model, data)

            # Calculate loss, backpropagation and update weights
            batch_loss = calculateLoss(output, target, lossFunction, optimizer)

            calculateBackpropagationStep(optimizer, batch_loss)
            
            train_loss += batch_loss.item() # Add the loss to the total loss
            
            # progress_bar_train.update(1)
        
        # close the progress bar 
        # progress_bar_train.close()
        
        # average loss per epoch
        train_losses.append(train_loss / len(train_loader))



        # evaluate the model in each epoch if evaluation is setted
        if(haveToEvaluate):
            model.eval()
            # avoid change gradients
            with torch.no_grad():     
                # progress_bar_eval = tqdm(total=len(eval_loader), desc='Processing eval batches', unit='batch', leave=False)
                
                valid_loss = 0
                
                # Cicles for each batch in eval
                for data, target in eval_loader:
                    
                    # Get predictions
                    output = getPrediction(model, data)           

                    # Calculate loss, backpropagation and update weights
                    batch_loss = calculateLoss(output, target, lossFunction, optimizer)
                    
                    valid_loss += batch_loss.item() # Add the loss to the total loss

                    # progress_bar_eval.update(1)
                
                # close the progress bar
                # progress_bar_eval.close()

                # average loss per epoch
                valid_losses.append(valid_loss / len(eval_loader))

            tqdm.write(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Valid Loss: {valid_losses[-1]}")
        else:
            tqdm.write(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}")
        
        # update the progress bar
        progress_bar_epochs.update(1)

        if(epoch % 5 == 0):
            saveModel(model, optimizer, models_path, epoch, train_losses, valid_losses)
            generateChart(charts_path, train_losses, valid_losses, epoch)
    
    # end the progress bar
    progress_bar_epochs.close()
     
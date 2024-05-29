import os
import torch
from torch import optim
from models.Model import CombinedImagesAutoencoder
from utils import utils
from arguments.arguments import parseTrainArguments
from train.train import train

# Código principal
if __name__ == '__main__':
    # get arguments
    args = parseTrainArguments()

    # load config yaml
    config = utils.loadConfig(args.config)

    # verify cuda
    useCuda = utils.setCudaIsAvailable(config, args.use_cuda)

    # get loaders of data
    train_loader, eval_loader, loaders_info = utils.getLoaders(args.training_data, config, useCuda, args.data_percentage_to_use, args.train_percentage_split,args.batch_size, args.image_size)
    
    # Create model
    model = CombinedImagesAutoencoder()
    if(useCuda):
        model = model.to('cuda')

    # create folder to save info of training
    folder_path, models_path, results_path, charts_path = utils.get_folder_for_training_model(args.folder_result, config)

    # get epochs
    epochs = utils.getEpochs(config, args.epochs)

    # get learning rate
    learning_rate = utils.getLearningRate(config, args.learning_rate)

    # Save info model in txt before training  
    model_info_path = os.path.join(models_path,'model_info.txt')
    with open(model_info_path, 'w') as file:
        file.write("INFO OF MODEL:\n")
        file.write(str(model))
        file.write("\n\nINFO OF TRAINING:\n")
        file.write("Percentage Data Use: "+str(loaders_info["data_percentage"])+"\n")
        file.write("Percentage Data for training (of total): "+str(loaders_info["data_percentage_train"])+"\n")
        file.write("Batch size: "+ str(loaders_info["batch_size"])+"\n")
        file.write("Image size: "+ str(loaders_info["image_size"])+"\n")
        file.write("Learning rate: "+str(learning_rate)+"\n")
        file.write("Epochs: "+ str(epochs) +"\n")
        
        

    # Train the model
    train(model, train_loader, eval_loader, epochs=epochs, charts_path=charts_path, learning_rate=learning_rate)

    # Guardar el modelo
    torch.save(model.state_dict(), os.path.join(models_path,'model.pth'))
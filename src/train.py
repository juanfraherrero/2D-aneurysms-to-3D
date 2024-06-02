import os
import torch
import torch.autograd.profiler as profiler
from models.Model import CombinedImagesAutoencoder
from utils import utils
from utils.warmUpModel import warmUpModel 
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

    # create folder to save info of training
    folder_path, models_path, results_path, charts_path = utils.get_folder_for_training_model(args.folder_result, config)

    # get epochs
    epochs = utils.getEpochs(config, args.epochs)

    # get learning rate
    learning_rate = utils.getLearningRate(config, args.learning_rate)

    # get if running in colab
    isRunningInColab = utils.getColab(config, args.colab)

    # get learning rate
    resume_training = utils.getResumeEpoch(config, args.resume_training)
    
    # get checkpoint path
    checkpoint_path = utils.getCheckpointPath(config, args.checkpoint_path)
    
    # Profiling
    profiling = utils.getProfiling(config, args.profiling)

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

    if profiling:
        print("Porfiling")
        if(useCuda):
            model.to('cuda')
        
        input = torch.rand(128, 500).cuda()
        mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

        # warm up
        warmUpModel(model, train_loader)

        with profiler.profile(record_shapes=True) as prof:
            train(model, train_loader, eval_loader, epochs, models_path, charts_path, learning_rate, isRunningInColab, resume_training, checkpoint_path, useCuda)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        exit()

    # Train the model
    train(model, train_loader, eval_loader, epochs=epochs, models_path=models_path, charts_path=charts_path, learning_rate=learning_rate, isRunningInColab=isRunningInColab, resume_training=resume_training, checkpoint_path=checkpoint_path, useCuda=useCuda)

    # Guardar el modelo
    torch.save(model.state_dict(), os.path.join(models_path,'model.pth'))
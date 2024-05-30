import os
import yaml
import torch
from data.loaders import ImagesDataset
from torch.utils.data import DataLoader, random_split
from data.customCollate import custom_collate_fn

def get_folder_for_training_model(folder_result_path, config):
  """
    Creates folders and subfolders for results and models
    returns the paths to the folders
  """
  if(folder_result_path is not None):
    full_path =  folder_result_path
  else:
    full_path = config["folder_result_path"]
  
  
  # get quantit of folders
  if(not os.path.exists(full_path)):
    quantity_result_folders = 0
  else:
    quantity_result_folders = len([os.path.join(full_path,path) for path in os.listdir(full_path) if os.path.isdir(os.path.join(full_path,path))])

      

  folder_path = os.path.join(full_path,"train"+str(quantity_result_folders))
  models_path = os.path.join(folder_path,"model")
  results_path = os.path.join(folder_path,"results")
  charts_path = os.path.join(folder_path,"charts")
  
  os.makedirs(models_path, exist_ok=True)
  os.makedirs(results_path, exist_ok=True)
  os.makedirs(charts_path, exist_ok=True)

  return folder_path, models_path, results_path, charts_path

def loadConfig(config_path):
  """
    Load the config file and return the dictionary
  """
  if (config_path is not None):
    config_path = config_path 
  else:
    #default config path
    config_path = 'src/config_train.yaml' 
  
  # load config yaml
  with open(config_path, 'r') as file:
    return yaml.safe_load(file)

def setCudaIsAvailable(config, use_cuda):
  """
    Set if cuda is available
    Print if cuda is available
  """
  if(use_cuda or config["useCuda"]):
    useCuda = torch.cuda.is_available()
  else:
    useCuda = False
      
  print ("Using Cuda: ",useCuda)
  return useCuda
  

def getLoaders(training_path, config, useCuda, data_percentage_to_use,train_percentage_split, batch_size, image_size):
  """
    Get the loaders for training and validation
    Print the size of the datasets
  """
  # get folder path
  folder_path = getFolderPath(training_path, config)

  # get image size
  image_size = getImageSize(config, image_size)

  # create dataset
  dataset = ImagesDataset(folder_path=folder_path, size=image_size, useCuda=useCuda)

  #get percentage of data to use in training and validation  
  train_percentage, valid_percentage, trash_percentage, total_size = getDatasetSizePercentage(dataset, config, data_percentage_to_use, train_percentage_split)

  print(train_percentage, valid_percentage, trash_percentage, total_size)
  # get datasets splitted
  train_dataset, eval_dataset, trash_dataset = random_split(dataset, [train_percentage, valid_percentage, trash_percentage])
  
  print(f"Original size: {len(dataset)} \nTotal size: {total_size} \nTrain size: {len(train_dataset)} \nValidation size: {len(eval_dataset)}")

  bs = getBatchSize(config, batch_size)

  train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=custom_collate_fn)
  eval_loader = DataLoader(eval_dataset, batch_size=bs, shuffle=False, collate_fn=custom_collate_fn)
  loaders_info = {
    "data_percentage": 1 - trash_percentage,
    "data_percentage_train": train_percentage,
    "batch_size": bs,
    "image_size": image_size[0]
  }
  return train_loader, eval_loader, loaders_info

def getDatasetSizePercentage(dataset, config, data_percentage_to_use,train_percentage_split):
  """
    Get the percentage of the datasets to split by
  """
  data_percentage = getDataPercentage(data_percentage_to_use, config)
  data_train_percentage = getDataTrainPercentage(train_percentage_split, config)
  data_eval_percentage = 1 - data_train_percentage
  
  train_percentage = data_percentage * data_train_percentage
  valid_percentage = data_percentage - train_percentage
  trash_percentage = 1 - data_percentage
  total_size = int(len(dataset)*data_percentage)
  
  return train_percentage, valid_percentage, trash_percentage, total_size

""" 
    functions to get the values of the arguments or config
"""
def getFolderPath(training_path, config):
  """
    Get the folder path
  """
  if(training_path is not None):
    folder_path = training_path
  else:
    folder_path = config["folder_data_path"]
  return folder_path

def getBatchSize(config, batch_size):
  """
    Get the batch size
  """
  if (batch_size is not None):
    bs = batch_size
  else:
    bs = config["batch_size"]
  return bs

def getDataPercentage(data_percentage_to_use, config):
  """
    Get the percentage of data to use
  """
  if(data_percentage_to_use is not None):
    data_percentage = data_percentage_to_use
  else:
    data_percentage = config["data_percentage_to_use"]
  return data_percentage

def getDataTrainPercentage(train_percentage_split, config):
  """
    Get the percentage of data to use in training
  """
  if(train_percentage_split is not None):
    data_train_percentage = train_percentage_split
  else:
    data_train_percentage = config["train_percentage_split"]
  return data_train_percentage

def getImageSize(config, image_size):
  """
    Get the image size
  """
  if(image_size is not None):
    image_size = [image_size, image_size]
  else:
    image_size = [config["image_size"], config["image_size"]]
  return image_size

def getEpochs(config, epochs):
  """
    Get the number of epochs
  """
  if(epochs is not None):
    epochs = epochs
  else:
    epochs = config["epochs"]
  return epochs

def getLearningRate(config, learning_rate):
  """
    Get the learning rate
  """
  if(learning_rate is not None):
    learning_rate = learning_rate
  else:
    learning_rate = config["learning_rate"]
  return learning_rate

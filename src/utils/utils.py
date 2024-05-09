import os
def get_folder_for_training_model():
  """
    Create a folder with the name folder_name in the results folder
    returns the path to the folder
  """
  full_path = os.path.join(os.getcwd(), "model_training")
  
  # load all folders of each dataset
  quantity_result_folders = len([os.path.join(full_path,path) for path in os.listdir(full_path) if os.path.isdir(os.path.join(full_path,path))])
      

  folder_path = os.path.join(full_path,"train"+str(quantity_result_folders))
  models_path = os.path.join(folder_path,"model")
  results_path = os.path.join(folder_path,"results")
  charts_path = os.path.join(folder_path,"charts")
  
  os.makedirs(models_path, exist_ok=True)
  os.makedirs(results_path, exist_ok=True)
  os.makedirs(charts_path, exist_ok=True)

  return folder_path, models_path, results_path, charts_path
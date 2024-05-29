import argparse              
def parseTrainArguments():
  # add -td argument to specify the training data directory
  # add -c argument to specify the configuration file
  parser = argparse.ArgumentParser(description='Script to train model')
  parser.add_argument('-td', '--training_data', type=str, help='Directory where is the training data')
  parser.add_argument('-c', '--config', type=str, help='file of configuration')
  parser.add_argument('-fr', '--folder_result', type=str, help='folder to save the results')
  parser.add_argument('-dptu', '--data_percentage_to_use', type=float, help='percentage of data to use of total data size (testing purposes)')
  parser.add_argument('-tps', '--train_percentage_split', type=float, help='percentage of data to use in training, else for validation')
  parser.add_argument('-bs', '--batch_size', type=int, help='batch size')
  parser.add_argument('-is', '--image_size', type=int, help='size of the image')
  parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
  parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate')
  parser.add_argument('-uc', '--use_cuda', type=bool, help='use cuda')
  return parser.parse_args()


import torch

def loadCheckpoint(model, checkpoint_path):
  """
    Load the checkpoint
  """
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer_checkpoint = checkpoint['optimizer_state_dict']
  start_epoch = checkpoint['epoch']
  train_losses = checkpoint['train_loss']
  valid_losses = checkpoint['valid_loss']
  
  return model, optimizer_checkpoint, start_epoch, train_losses, valid_losses
def calculateLoss(output, target, lossFunction, optimizer):
  """
    Calculate the loss, backpropagation and update weights
  """
  # Cicle over the batch size and calculate loss
  batch_loss = 0

  for idx in range(len(target)):
    # Add one dimension to target and ouput --- here can be parallelized¿?
    tg = target[idx].unsqueeze(0) 
    out = output[idx].unsqueeze(0)
    batch_loss += lossFunction(tg, out)
  
  # Average loss in batch
  batch_loss /= len(target) 

  return batch_loss
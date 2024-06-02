def calculateBackpropagationStep (optimizer, batch_loss):
  # Calculate gradients
  batch_loss.backward() 
  
  # Update weights
  optimizer.step() 

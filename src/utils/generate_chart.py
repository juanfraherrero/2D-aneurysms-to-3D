import os 
import matplotlib.pyplot as plt

def generateChart(charts_path, train_losses, valid_losses,epochs):
    """
        Generate a chart with the train and valid losses
    """

    # plot the losses and save it in dir
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if(valid_losses != []):
      plt.plot(valid_losses, label='Valid Loss')
      plt.title('Training and Validation Loss per Epoch')
    else:
      plt.title('Training Loss per Epoch')    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(charts_path, f'train_valid_losses_epoch_{epochs}.png'))
 
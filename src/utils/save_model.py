import torch 

def saveModel(model, optimizer, models_path, epoch, train_losses, valid_losses):
    """
        Save the model in the models_path with the epoch number
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_losses,
        'valid_loss': valid_losses
    }, f"{models_path}/model_epoch{epoch}.tar")
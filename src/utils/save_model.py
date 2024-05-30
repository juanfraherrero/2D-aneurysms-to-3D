import torch 

def saveModel(model, models_path, epoch):
    """
        Save the model in the models_path with the epoch number
    """
    torch.save(model.state_dict(), f"{models_path}/model_epoch{epoch}.pt")
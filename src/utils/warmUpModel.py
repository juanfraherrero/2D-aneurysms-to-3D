def warmUpModel(model, train_loader):
  for data, _ in train_loader:
      _ = model(data)
      break
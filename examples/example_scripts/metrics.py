import torch

def accuracy_score(outputs, targets):
    return torch.sum(torch.argmax(outputs, dim=1) == targets).item() / len(outputs)

def rmse_score(outputs, targets):
    return torch.sqrt(torch.mean((targets-outputs)**2)).item()
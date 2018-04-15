import torch

def save_model(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path):
    model.load_state_dict(path)
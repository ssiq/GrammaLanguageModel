import torch

def save_model(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path):
    model.load_state_dict(path)

def mask_softmax(logit, mask):
    logit = logit * mask
    logit_max, _ = torch.max(logit, dim=-1, keepdim=True)
    logit = logit - logit_max
    logit_exp = torch.exp(logit) * mask
    softmax = logit_exp/torch.sum(logit_exp, dim=-1, keepdim=True)
    return softmax

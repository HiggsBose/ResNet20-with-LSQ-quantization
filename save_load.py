import torch


def save(model):
    torch.save(model, 'model.pt')
    return None

def load(path):
    model = torch.load(path)
    return model
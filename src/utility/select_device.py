import torch


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
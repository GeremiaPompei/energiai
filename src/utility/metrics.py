import torch

eps = 1e-17


def accuracy(y, p):
    return (y - p == 0).to(torch.float64).mean().item()


def precision(y, p):
    tp = torch.logical_and(y == 1, p == 1).to(torch.float64).sum().item()
    fp = torch.logical_and(y == 0, p == 1).to(torch.float64).sum().item()
    return tp / (tp + fp + eps)


def recall(y, p):
    tp = torch.logical_and(y == 1, p == 1).to(torch.float64).sum().item()
    fn = torch.logical_and(y == 1, p == 0).to(torch.float64).sum().item()
    return tp / (tp + fn + eps)


def f1_score(y, p):
    prec = precision(y, p)
    rec = recall(y, p)
    return 2 * prec * rec / (prec + rec + eps)


def compute_scores(y, p):
    return {callback.__name__: callback(y, p) for callback in [
        accuracy, precision, recall, f1_score
    ]}

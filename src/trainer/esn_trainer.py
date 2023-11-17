import torch
from torch import nn

from src.model.esn import ESN
from src.utility import log


class ESNTrainer:
    def __init__(self, model: ESN, tr_loader, ts_loader, device='cpu'):
        self.model = model
        self.tr_loader = tr_loader
        self.ts_loader = ts_loader
        self.device = device

    def __call__(self):
        tr_loss, ts_loss, ts_acc = 0, 0, 0
        criterion = torch.nn.MSELoss()
        self.model.train()
        for batch_idx, (data, _) in enumerate(self.tr_loader):
            data = data.to(self.device)
            A, B = self.model.compute_ab(data[:, 1:], data[:, :-1])
            if self.model.A is not None:
                A += self.model.A
                B += self.model.B
            self.model.fit_from_ab(A, B)
            x, y = data[:, 1:], data[:, :-1]
            p = self.model(x)
            tr_loss += criterion(y, p).item()
        tr_loss = tr_loss / len(self.tr_loader)

        for batch_idx, (data, _) in enumerate(self.tr_loader):
            data = data.to(self.device)
            x, y = data[:, 1:], data[:, :-1]
            self.model.register_std(x, y)

        self.model.eval()
        for batch_idx, (data, labels) in enumerate(self.ts_loader):
            data = data.to(self.device)
            x, y = data[:, 1:], data[:, :-1]
            p, _, _ = self.model.predict(x, y)
            labels = labels[:, -p.shape[1]:]
            ts_loss += criterion(labels, p).item()
            ts_acc += (labels - p == 0).to(torch.float64).mean()
        ts_loss = ts_loss / len(self.ts_loader)
        ts_acc = ts_acc / len(self.ts_loader)
        ts_acc = f'{round(ts_acc.item() * 100, 2)}%'

        log.info(
            f'training loss: {tr_loss}, test loss: {ts_loss}, test accuracy: {ts_acc}')

        return tr_loss, ts_loss

import torch
from torch import nn
from src.utility import log


class LSTMTrainer:
    def __init__(self, model, tr_loader, ts_loader, device='cpu'):
        self.model = model
        self.tr_loader = tr_loader
        self.ts_loader = ts_loader
        self.device = device

    def __call__(self, epochs: int = 10, lr: float = 1e-3):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        tr_loss, ts_loss = 0, 0
        for epoch in range(epochs):
            tr_loss, ts_loss, ts_acc = 0, 0, 0

            self.model.train()
            self.model.reset()
            for batch_idx, (data, _) in enumerate(self.tr_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                x, y = data[:, 1:], data[:, :-1]
                p = self.model(x, y)
                loss = criterion(y, p)
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
            tr_loss = tr_loss / len(self.tr_loader)

            self.model.eval()
            for batch_idx, (data, labels) in enumerate(self.ts_loader):
                data = data.to(self.device)
                x, y = data[:, 1:], data[:, :-1]
                p, _, _ = self.model(x, y)
                labels = labels[:, -p.shape[1]:]
                ts_loss += criterion(labels, p).item()
                ts_acc += (labels - p == 0).to(torch.float64).mean()
            ts_loss = ts_loss / len(self.ts_loader)
            ts_acc = ts_acc / len(self.ts_loader)
            ts_acc = f'{round(ts_acc.item() * 100, 2)}%'

            log.info(f'Epoch {epoch + 1}/{epochs} => training loss: {tr_loss}, test loss: {ts_loss}, test accuracy: {ts_acc}')
        return tr_loss, ts_loss

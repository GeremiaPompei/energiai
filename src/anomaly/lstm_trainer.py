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
            tr_loss, ts_loss = 0, 0

            self.model.train()
            for batch_idx, (x, y) in enumerate(self.tr_loader):
                optimizer.zero_grad()
                p = self.model(x.to(self.device))
                loss = criterion(y.to(self.device), p)
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
            tr_loss = tr_loss / len(self.tr_loader)

            self.model.eval()
            for batch_idx, (x, y) in enumerate(self.ts_loader):
                p = self.model(x.to(self.device))
                loss = criterion()
                ts_loss += loss.item(y.to(self.device), p)
            ts_loss = ts_loss / len(self.ts_loader)

            log.info(f'Epoch {epoch + 1}/{epochs} => training loss: {tr_loss}, test loss: {ts_loss}')
        return tr_loss, ts_loss

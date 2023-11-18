import torch
from src.trainer.trainer import Trainer


class LSTMTrainer(Trainer):

    def train_model(self, criterion, epochs: int = 10, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        tr_loss = 0
        for epoch in range(epochs):
            tr_loss = 0
            self.model.train()
            self.model.reset()
            for batch_idx, (data, _) in enumerate(self.tr_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                x, y = data[:, 1:], data[:, :-1]
                p = self.model(x)
                loss = criterion(y, p)
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
            tr_loss = tr_loss / len(self.tr_loader)
        self.model.eval()
        return tr_loss

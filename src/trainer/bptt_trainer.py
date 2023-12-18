from typing import Callable

import torch
from src.trainer.trainer import Trainer


class BPTTTrainer(Trainer):

    def __init__(self, *args, callback: Callable = None, **kwargs):
        super(BPTTTrainer, self).__init__(*args, **kwargs)
        self.callback = callback

    def train_model(self, *args, epochs: int = 10, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.99,
                    weight_decay: float = 0, **kwargs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
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
                loss = self.criterion(y, p)
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
            tr_loss = tr_loss / len(self.tr_loader)
            if self.callback is not None:
                self.callback(self)
        self.model.eval()
        return tr_loss

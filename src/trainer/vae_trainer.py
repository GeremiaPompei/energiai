import torch
from torch import nn
from tqdm import tqdm


class VAETrainer:
    def __init__(self, model, tr_loader, ts_loader, device='cpu'):
        self.model = model
        self.tr_loader = tr_loader
        self.ts_loader = ts_loader
        self.device = device

    def __loss_function__(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

    def __call__(self, epochs: int = 10, lr: float = 1e-3):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        pbar = tqdm(range(epochs), desc='training')
        tr_loss, ts_loss = 0, 0
        for epoch in pbar:
            tr_loss, ts_loss = 0, 0

            self.model.train()
            for batch_idx, (x, y) in enumerate(self.tr_loader):
                optimizer.zero_grad()
                x = x.to(self.device)
                x_hat, mean, log_var = self.model(x)
                loss = self.__loss_function__(x, x_hat, mean, log_var)
                tr_loss += loss.item()
                loss.backward()
                optimizer.step()
            tr_loss = tr_loss / len(self.tr_loader)

            self.model.eval()
            for batch_idx, (x, y) in enumerate(self.ts_loader):
                x = x.to(self.device)
                x_hat, mean, log_var = self.model(x)
                loss = (x - x_hat).pow(2).mean()
                ts_loss += loss.item()
            ts_loss = ts_loss / len(self.ts_loader)

            pbar.write(f'training loss: {tr_loss}, test loss: {ts_loss}')
        return tr_loss, ts_loss

import torch
from torch import nn
from tqdm import tqdm


class VAETrainer:
    def __init__(self, model, lr: float = 1e-3, device='cpu'):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device

    def __loss_function__(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    def __call__(self, train_loader, epochs: int = 10):
        self.model.train()
        loss = 0
        pbar = tqdm(range(epochs), desc='training...')
        for epoch in pbar:
            loss = 0
            self.model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                self.optimizer.zero_grad()

                x_hat, mean, log_var = self.model(x)
                loss = self.__loss_function__(x, x_hat, mean, log_var)

                loss += loss.item()

                loss.backward()
                self.optimizer.step()
            pbar.write(f'training loss: {round(loss / len(train_loader), 4)}')
        self.model.eval()
        return loss

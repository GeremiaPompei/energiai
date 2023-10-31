import pandas as pd
import torch


class SifimDataset(torch.utils.data.Dataset):

    def __init__(self, timesteps=10, start=0, end=1):
        self.dataset = torch.from_numpy(pd.read_csv('dataset/cleaned/dataset.csv').to_numpy()[:, 1:])
        n_examples = int(self.dataset.shape[0] / timesteps)
        self.dataset = self.dataset[:n_examples * timesteps].view(n_examples, timesteps, self.dataset.shape[-1])
        init_size, end_size = int(start * self.dataset.shape[0]), int(end * self.dataset.shape[0])
        self.dataset = self.dataset[init_size:end_size]

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        X = self.dataset[idx]
        return X, 1

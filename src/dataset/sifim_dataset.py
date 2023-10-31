import os

import pandas as pd
import torch


class SifimDataset(torch.utils.data.Dataset):

    def __init__(self, dir='dataset/cleaned/', timesteps=10, start=0, end=1):
        datasets = []

        for filename in os.listdir(dir):
            if not filename.startswith('.'):
                dataset = torch.from_numpy(pd.read_csv(f'{dir}/{filename}').to_numpy()[:, 1:]).to(torch.float64)
                n_examples = int(dataset.shape[0] / timesteps)
                dataset = dataset[:n_examples * timesteps].view(n_examples, timesteps, dataset.shape[-1])
                init_size, end_size = int(start * dataset.shape[0]), int(end * dataset.shape[0])
                datasets.append(dataset[init_size:end_size])

        self.dataset = torch.cat(datasets)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        X = self.dataset[idx]
        return X, 1

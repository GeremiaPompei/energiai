import os

import pandas as pd
import torch


class SifimDataset(torch.utils.data.Dataset):

    def __init__(self, dir='dataset/cleaned/', timesteps=300, start=0, end=1, test=False, noise=0.1):
        datasets = []

        for filename in os.listdir(dir):
            if not filename.startswith('.'):
                dataset = torch.from_numpy(pd.read_csv(f'{dir}/{filename}').to_numpy()[:, 1:]).to(torch.float64)
                n_examples = int(dataset.shape[0] / timesteps)
                dataset = dataset[:n_examples * timesteps].view(n_examples, timesteps, dataset.shape[-1])
                init_size, end_size = int(start * dataset.shape[0]), int(end * dataset.shape[0])
                datasets.append(dataset[init_size:end_size])

        self.x = torch.cat(datasets)
        self.y = torch.zeros_like(self.x)
        if test:
            half_ts = timesteps // 2
            n_features = self.x.shape[-1]

            anomaly = torch.randn(self.x.shape[0], half_ts, n_features) * noise
            self.x[:, half_ts:] = self.x[:, half_ts:] + anomaly
            self.y[:, half_ts:] = 1
        self.y = self.y.squeeze().to(torch.int64)

        self.x = self.x.to(torch.float64)
        self.y = self.y.to(torch.float64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

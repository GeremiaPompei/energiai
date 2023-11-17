import os

import pandas as pd
import torch


class SifimDataset(torch.utils.data.Dataset):

    def __init__(self, dir='dataset/cleaned/', timesteps=100, start=0, end=1, test=False):
        datasets = []

        for filename in os.listdir(dir):
            if not filename.startswith('.'):
                dataset = torch.from_numpy(pd.read_csv(f'{dir}/{filename}').to_numpy()[:, 1:]).to(torch.float64)
                n_examples = int(dataset.shape[0] / timesteps)
                dataset = dataset[:n_examples * timesteps].view(n_examples, timesteps, dataset.shape[-1])
                init_size, end_size = int(start * dataset.shape[0]), int(end * dataset.shape[0])
                datasets.append(dataset[init_size:end_size])

        self.x = torch.cat(datasets)
        self.y = torch.zeros(self.x.shape[0], self.x.shape[1], 1)
        if test:
            half_ex = self.x.shape[0] // 2
            half_ts = timesteps // 2
            n_features = self.x.shape[-1]

            drift = torch.rand(half_ex).repeat(n_features, half_ts, 1).transpose(0, 2)
            self.x[:half_ex, half_ts:] = (self.x[:half_ex, half_ts:] + drift) % 1
            self.y[:half_ex, half_ts:] = 1

            anomaly = torch.rand(half_ex, half_ts, n_features)
            self.x[half_ex:, half_ts:] = (self.x[half_ex:, half_ts:] + anomaly) % 1
            self.y[half_ex:, half_ts:] = 2
        y = self.y.squeeze().to(torch.int64)
        self.y = torch.nn.functional.one_hot(y, 3)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

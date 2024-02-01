import json
import os

import numpy as np
import pandas as pd
import torch

from src.utility import fix_seed


def create_sifim_datasets(dir='dataset/cleaned/', timesteps=300, vl_perc=0.2, ts_perc=0.2, noise=0.2, seed=0):
    fix_seed(seed=seed)
    tr_set, vl_set, ts_set = [], [], []

    norm_path = 'dataset/normalization/scale_factors.json'
    with open(norm_path) as fp:
        sf = json.load(fp)
        noise *= np.array(sf['std'])

    for filename in os.listdir(dir):
        if not filename.startswith('.'):
            dataset = torch.from_numpy(pd.read_csv(f'{dir}/{filename}').to_numpy()[:, 1:]).to(torch.float64)
            n_examples = int(dataset.shape[0] / timesteps)
            dataset = dataset[:n_examples * timesteps].view(n_examples, timesteps, dataset.shape[-1])
            dataset = dataset[torch.randperm(n_examples)]
            vl_size, ts_size = int(vl_perc * dataset.shape[0]), int(ts_perc * dataset.shape[0])

            tr_set.append(dataset[:- vl_size - ts_size])
            vl_set.append(dataset[- vl_size - ts_size:-ts_size])
            ts_set.append(dataset[-ts_size:])

    datasets = []
    for dataset, test in [(tr_set, False), (vl_set, True), (ts_set, True)]:
        x = torch.cat(dataset)
        y = torch.zeros_like(x)
        if test:
            half_ts = timesteps // 2
            n_features = x.shape[-1]

            anomaly = torch.randn(x.shape[0], half_ts, n_features) * noise
            x[:, half_ts:] = x[:, half_ts:] + anomaly
            y[:, half_ts:] = 1
        y = y.squeeze().to(torch.int64)

        datasets.append(SifimDataset(x.to(torch.float64), y.to(torch.float64)))

    return tuple(datasets)


class SifimDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

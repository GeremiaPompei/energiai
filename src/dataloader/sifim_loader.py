import os

import pandas as pd
import torch


class SifimLoader(torch.utils.data.Dataset):

    def __init__(self):
        self.dataset = torch.cat(
            [torch.from_numpy(pd.read_csv(fn).to_numpy()) for fn in os.listdir('dataset/cleaned')],
            dim=1
        )

    def __len__(self):
        return self.dataset.shape[1]

    def __getitem__(self, idx):
        X = self.dataset[idx]
        return X, torch.IntTensor(1)

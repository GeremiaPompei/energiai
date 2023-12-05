import json
import os

import numpy as np
import pandas as pd
from functools import reduce

from src.utility.scaler import scale
from src.utility.constants import sifim_features


def preprocessing_pipeline(raw_dir: str = 'dataset/raw', output_dir: str = 'dataset'):
    columns_switchoff = ['corrente_di_neutro', 'corrente_di_sistema', 'corrente_fase_1', 'corrente_fase_2',
                         'corrente_fase_3']
    threshold_current = 0.1
    datasets = {}
    for filename in os.listdir(raw_dir):
        # import
        df_raw = pd.read_csv(f'{raw_dir}/{filename}').sort_values('timestamp')

        # cleaning
        df = df_raw[sifim_features]
        df = df[reduce(lambda x, y: x & y, [
                       df[cs] > threshold_current for cs in columns_switchoff])]
        datasets[filename] = df.to_numpy()
    X = np.concatenate(list(datasets.values()))
    X = X.T
    # standardization
    X_mean, X_std = X.mean(0), X.std(0)
    X = (X - X_mean) / X_std
    X_min, X_max = X.min(0), X.max(0)
    dataset = X.T

    pd.DataFrame(dataset).to_csv('dataset/cleaned/dataset.csv')
    pd.DataFrame(X_mean).to_csv('dataset/normalization/mean.csv')
    pd.DataFrame(X_std).to_csv('dataset/normalization/std.csv')
    pd.DataFrame(X_min).to_csv('dataset/normalization/min.csv')
    pd.DataFrame(X_max).to_csv('dataset/normalization/max.csv')

    # cleaned_subdir, normalization_subdir = f'{output_dir}/cleaned', f'{output_dir}/normalization'
    # for path in [cleaned_subdir, normalization_subdir]:
    #     if not os.path.exists(path):
    #         os.mkdir(path)

    # norm_path = f'{normalization_subdir}/scale_factors1.json'
    # with open(norm_path, 'w') as fp:
    #     json.dump(
    #         dict(
    #             mean=X_mean.tolist(),
    #             std=X_std.tolist(),
    #             max=X_max.tolist(),
    #             min=X_min.tolist(),
    #         ), fp
    #     )

    # for filename, dataset in datasets.items():
    #     dataset = scale(dataset, norm_path=norm_path)
    #     pd.DataFrame(dataset).to_csv(f'{cleaned_subdir}/{filename}')

import json
import os

import numpy as np
import pandas as pd
from functools import reduce

columns_blacklist = ['timestamp', 'id', 'sub_id', 'contatore_di_installazione', 'contatore_di_misura',
                     'numero_seriale']

columns_switchoff = ['corrente_di_neutro', 'corrente_di_sistema', 'corrente_fase_1', 'corrente_fase_2',
                     'corrente_fase_3']
threshold_current = 0.1


def preprocessing_pipeline(raw_dir: str = 'dataset/raw', output_dir: str = 'dataset'):
    datasets = {}
    for filename in os.listdir(raw_dir):
        # import
        df_raw = pd.read_csv(f'{raw_dir}/{filename}')

        # cleaning
        df = df_raw.drop(columns=columns_blacklist)
        df = df[reduce(lambda x, y: x & y, [df[cs] > threshold_current for cs in columns_switchoff])]
        datasets[filename] = df.to_numpy()[::-1]
    X = np.concatenate(list(datasets.values()))

    # standardization
    X_mean, X_std = X.mean(), X.std()
    X = (X - X_mean) / X_std
    X_min, X_max = X.min(), X.max()

    cleaned_subdir, normalization_subdir = f'{output_dir}/cleaned', f'{output_dir}/normalization'
    for path in [cleaned_subdir, normalization_subdir]:
        if not os.path.exists(path):
            os.mkdir(path)

    with open(f'{normalization_subdir}/scale_factors.json', 'w') as fp:
        json.dump(
            dict(
                mean=X_mean,
                std=X_std,
                max=X_max,
                min=X_min,
            ), fp
        )

    for filename, dataset in datasets.items():
        dataset = (dataset - X_mean) / X_std
        dataset = (dataset - X_min) / (X_max - X_min)
        pd.DataFrame(dataset).to_csv(f'{cleaned_subdir}/{filename}')

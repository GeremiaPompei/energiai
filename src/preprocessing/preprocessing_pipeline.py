import os

import numpy as np
import pandas as pd
from functools import reduce

eps = 1e-17

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
    X_mean, X_std = X.mean(0), X.std(0)
    X_std[X_std == 0] = eps
    X = (X - X_mean) / X_std
    # normalization (min-max scaler)
    X_min, X_max = X.min(0), X.max(0)
    X_max[X_max == X_min] += eps
    # X = (X - X_min) / (X_max - X_min)

    cleaned_subdir, normalization_subdir = f'{output_dir}/cleaned', f'{output_dir}/normalization'
    for path in [cleaned_subdir, normalization_subdir]:
        if not os.path.exists(path):
            os.mkdir(path)

    pd.DataFrame(X_mean).to_csv(f'{normalization_subdir}/mean.csv')
    pd.DataFrame(X_std).to_csv(f'{normalization_subdir}/std.csv')
    pd.DataFrame(X_min).to_csv(f'{normalization_subdir}/min.csv')
    pd.DataFrame(X_max).to_csv(f'{normalization_subdir}/max.csv')

    for filename, dataset in datasets.items():
        dataset = (dataset - X_mean) / X_std
        dataset = (dataset - X_min) / (X_max - X_min)
        pd.DataFrame(dataset).to_csv(f'{cleaned_subdir}/{filename}')

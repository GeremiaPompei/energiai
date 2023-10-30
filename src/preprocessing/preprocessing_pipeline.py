import numpy as np
import pandas as pd
from functools import reduce

columns_blacklist = ['timestamp', 'id', 'sub_id', 'contatore_di_installazione', 'contatore_di_misura',
                     'numero_seriale']

columns_switchoff = ['corrente_di_neutro', 'corrente_di_sistema', 'corrente_fase_1', 'corrente_fase_2',
                     'corrente_fase_3']
threshold_current = 0.1


def preprocessing_pipeline(filenames: str = ['1306-1406']):
    X = []
    for filename in filenames:
        # import
        df_raw = pd.read_csv(f'dataset/raw/{filename}.csv')

        # cleaning
        df = df_raw.drop(columns=columns_blacklist)
        df = df[reduce(lambda x, y: x & y, [df[cs] > threshold_current for cs in columns_switchoff])]
        X.append(df.to_numpy())
    X = np.concatenate(X)

    # normalization
    X_scaled = X.T
    # standardization
    X_mean, X_std = X_scaled.mean(0), X_scaled.std(0)
    X_scaled = (X_scaled - X_mean) / X_std
    # normalization (min-max scaler)
    X_min, X_max = X_scaled.min(0), X_scaled.max(0)
    X_scaled = (X_scaled - X_min) / (X_max - X_min)
    dataset = X_scaled.T

    pd.DataFrame(dataset).to_csv('dataset/cleaned/dataset.csv')
    pd.DataFrame(X_mean).to_csv('dataset/normalization/mean.csv')
    pd.DataFrame(X_std).to_csv('dataset/normalization/std.csv')
    pd.DataFrame(X_min).to_csv('dataset/normalization/min.csv')
    pd.DataFrame(X_max).to_csv('dataset/normalization/max.csv')

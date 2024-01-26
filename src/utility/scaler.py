import json
import numpy as np

eps = 1e-17


def scale(vec, norm_path='dataset/normalization/scale_factors.json'):
    with open(norm_path) as fp:
        sf = json.load(fp)

    return (vec - np.array(sf['mean'])) / (np.array(sf['std']) + eps)


def rescale(vec, feature, norm_path='dataset/normalization/scale_factors.json'):
    with open(norm_path) as fp:
        sf = json.load(fp)

    return vec * (np.array(sf['std'][feature]) + eps) + np.array(sf['mean'][feature])

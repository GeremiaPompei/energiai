import json


def scale(vec, norm_path='dataset/normalization/scale_factors.json'):
    with open(norm_path) as fp:
        sf = json.load(fp)

    vec = (vec - sf['mean']) / sf['std']
    return (vec - sf['min']) / (sf['max'] - sf['min'])


def rescale(vec, norm_path='dataset/normalization/scale_factors.json'):
    with open(norm_path) as fp:
        sf = json.load(fp)

    vec = vec * (sf['max'] - sf['min']) + sf['min']
    return vec * sf['std'] + sf['mean']

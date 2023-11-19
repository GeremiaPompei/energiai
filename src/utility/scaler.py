import json


def scale(vec, norm_path='dataset/normalization/scale_factors.json'):
    with open(norm_path) as fp:
        sf = json.load(fp)

    return ((vec - sf['mean']) / sf['std'] - sf['min']) / (sf['max'] - sf['min'])


def rescale(vec, norm_path='dataset/normalization/scale_factors.json'):
    with open(norm_path) as fp:
        sf = json.load(fp)

    return (vec * (sf['max'] - sf['min']) + sf['min']) * sf['std'] + sf['mean']

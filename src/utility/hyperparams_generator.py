import itertools


def gridsearch_generator(hyperparams_lists):
    return [dict(zip(hyperparams_lists.keys(), t)) for t in itertools.product(*hyperparams_lists.values())]


def randomsearch_generator(ranges, times=10):
    return [{k: callback(start, end) for k, (start, end, callback) in ranges.items()} for _ in range(times)]

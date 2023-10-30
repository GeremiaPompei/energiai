import itertools


def gridsearch_generator(hyperparams_lists):
    return [dict(zip(hyperparams_lists.keys(), t)) for t in itertools.product(*hyperparams_lists.values())]


def randomsearch_generator(ranges, times=10):
    return [{callback(start, end) for (start, end, callback) in ranges} for _ in range(times)]

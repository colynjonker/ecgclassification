import numpy as np


def linear_weight(labels):
    class_weight = {}
    for c in range(5):
        class_weight.update({c: len(labels) / float(np.count_nonzero(labels == c))})
    return class_weight

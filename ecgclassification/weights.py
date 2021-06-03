import numpy as np


def linear_weight(labels):
    class_weight = {}
    for c in range(5):
        class_weight.update({c: len(labels) / float(np.count_nonzero(labels == c))})
    return class_weight


def balanced_weight(labels):
    class_weight = {}
    unique, c = np.unique(labels, return_counts=True)
    counts = dict(zip(unique,c))
    print(counts)
    for c in range(5):
        class_weight.update({c: float(np.count_nonzero(labels == c)) / np.max(np.count_nonzero(labels == c))})
    return class_weight


def compute_sample_weights(labels):
    weights = []
    return weights


def multiclass_temporal_class_weights(targets, class_weights):
    s_weights = np.ones((targets.shape[0],))
    # if we are counting the classes, the weights do not exist yet!
    if class_weights is not None:
        for i in range(len(s_weights)):
            weight = 0.0
            for itarget, target in enumerate(targets[i]):
                weight += class_weights[itarget][int(round(target))]
            s_weights[i] = weight
    return s_weights


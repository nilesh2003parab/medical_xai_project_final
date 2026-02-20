import numpy as np


def stability_score(scores):
    return 1 - np.std(scores)
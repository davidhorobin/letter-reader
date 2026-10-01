import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    sum_exp = np.sum(e_x)
    return e_x / sum_exp

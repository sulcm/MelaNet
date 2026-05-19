import numpy as np


def softmax(x, axis=None) -> np.ndarray:
    _x = np.asarray(x)
    exp_x_shifted = np.exp(_x - np.max(_x, axis=axis, keepdims=True))
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)
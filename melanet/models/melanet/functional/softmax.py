import numpy as np


def softmax(x: np.ndarray, axis=None) -> np.ndarray:
    exp_x_shifted = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)
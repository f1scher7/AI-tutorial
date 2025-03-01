import numpy as np


def biases_initialization_func(bias_len, func_name):
    return biases_initialization_funcs[func_name](bias_len)


biases_initialization_funcs = {
    'relu': lambda bias_len: np.ones(bias_len) * 0.1,
    'sigmoid': lambda bias_len: np.random.randn(bias_len) * 0.01,
    'tanh': lambda bias_len: np.random.randn(bias_len) * 0.01,
    'softmax': lambda bias_len: np.ones(bias_len),
}
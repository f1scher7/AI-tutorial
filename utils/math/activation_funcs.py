import numpy as np


def activation_func(x, func_name):
    if func_name == 'softmax':
        exp_x = np.exp((x - np.max(x, axis=-1, keepdims=True))) # we do it to avoid the e ** 1000, e ** 70000
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True) # axis=-1 = the last dim
    else:
        return activation_funcs[func_name](x)

def activation_derivative_func(x, func_name):
    return activation_derivative_funcs[func_name](x)


activation_funcs = {
    'relu': lambda x: np.maximum(x, 0),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'tanh': lambda x: np.tanh(x)
}

activation_derivative_funcs = {
    'relu': lambda x: (x > 0).astype(float),
    'sigmoid': lambda x: x * (1 - x),
    'sigmoid_raw': lambda x: activation_funcs['sigmoid'](x) * (1 - activation_funcs['sigmoid'](x)),
    'tanh': lambda x: 1 - x * x,
    'tanh_raw': lambda x: 1 - np.tanh(x) ** 2
}
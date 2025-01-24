import numpy as np
from cryptography.hazmat.primitives.serialization import load_ssh_private_key
from paramiko.util import lookup_ssh_host_config


def training_loss_func(y_true, y_pred, training_loss_func_name):
    return training_loss_funcs[training_loss_func_name](y_true, y_pred)


def training_loss_derivative_func(y_true, y_pred, training_loss_func_name):
    return training_loss_derivative_funcs[training_loss_func_name](y_true, y_pred)


# for monitoring
training_loss_funcs = {
    'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
}

# for back propagation
training_loss_derivative_funcs = {
    'mse': lambda y_true, y_pred: -2 * (y_pred - y_true) / y_true.shape[0]
}
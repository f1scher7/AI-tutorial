import numpy as np


epsilon = 1e-15


# for monitoring
def training_loss_func(y_true, y_pred, batch_size=None, training_loss_func_name='mse'):
    if training_loss_func_name == 'binary_cross_entropy' and batch_size is not None:
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -1. / batch_size * np.sum(y_true * np.log(y_pred) + (1. - y_true) * np.log(1. - y_pred))
    elif training_loss_func_name == 'categorical_cross_entropy' and batch_size is not None:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -1. / batch_size * np.sum(y_true * np.log(y_pred))

    return training_loss_funcs[training_loss_func_name](y_true, y_pred)

training_loss_funcs = {
    'mse': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
}


# for back propagation
def training_loss_derivative_func(y_true, y_pred, batch_size=None, training_loss_func_name='mse'):
    if training_loss_func_name == 'binary_cross_entropy' and batch_size is not None:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_true))
    elif training_loss_func_name == 'categorical_cross_entropy' and batch_size is not None:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - y_true / y_pred

    return training_loss_derivative_funcs[training_loss_func_name](y_true, y_pred)

training_loss_derivative_funcs = {
    'mse': lambda y_true, y_pred: 2. * (y_pred - y_true) / y_true.shape[0]
}

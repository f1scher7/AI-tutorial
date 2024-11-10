import numpy as np


def sigmoid(X_f):
    return 1 / (1 + np.exp(-X_f))


def sigmoid_derivative(X_f):
    return X_f * (1 - X_f)

# We're using Min-Max normalization:
# when we have small spread in input data (age(18, 80));
# when we want to have a range for input data;
def min_max_normalization(X_f, normalization_range_f, min_val_f=None, max_val_f=None):
    if min_val_f is None:
        min_val_f = X_f.min(axis=0)
    if max_val_f is None:
        max_val_f = X_f.max(axis=0)

    min_range, max_range = normalization_range_f

    # For normalization range (0, 1)
    normalized_X = (X_f - min_val_f) / (max_val_f - min_val_f + 1e-10)

    # For normalization range (a, b); for the (0, 1) normalization_X = scaled_X
    scaled_X = normalized_X * (max_range - min_range) + min_range

    return scaled_X, min_val_f, max_val_f


# We're using Log normalization:
# when we have large spread in input data (price(1000$-1000000$));
# when we have large numbers in input data;
def log_normalization(X_f):
    X_f += 1e-10 # We're adding a small value to avoid log(0)
    return np.log(X_f)

def mean_squared_error(y_true_f, y_pred_f):
    return np.mean((y_true_f - y_pred_f) ** 2)
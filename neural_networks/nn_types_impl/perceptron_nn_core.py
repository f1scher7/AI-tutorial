import numpy as np

from time import perf_counter
from utils import math_utils
from utils import displaying_nn_utils


# Perceptrons are used for classification problems(mostly binary classification), typically use the sigmoid activation function
def train_perceptron_nn(X_f, y_f, epochs_f, learning_rate_f):
    np.random.seed(42)

    input_neurons = X_f.shape[1]
    output_neurons = y_f.shape[1]

    input_to_output_weights = np.random.uniform(low=-1, high=1, size=(input_neurons, output_neurons))
    bias_output_weights = np.random.uniform(low=-1, high=1, size=(output_neurons, output_neurons))

    final_nn_output = np.array([])
    mse_values = []

    start_time = perf_counter()

    for epoch in range(epochs_f):
        # Forward propagation
        output_pre_activation = np.dot(X_f, input_to_output_weights) + bias_output_weights
        output_activated = math_utils.sigmoid(output_pre_activation)

        mse_values.append(math_utils.mean_squared_error(y_f, output_activated))

        final_nn_output = output_activated

        # Back propagation
        output_error = y_f - output_activated
        output_gradient = output_error * math_utils.sigmoid_derivative(output_activated)

        input_to_output_weights += learning_rate_f * np.dot(X_f.T, output_gradient)
        bias_output_weights += learning_rate_f * np.sum(output_gradient, axis=0, keepdims=True)

    end_time = perf_counter()
    training_time = end_time - start_time

    displaying_nn_utils.print_result_nn(final_nn_output, epochs_f, training_time)
    displaying_nn_utils.plot_mse(mse_values)

    return input_to_output_weights, bias_output_weights


def predict_perceptron_nn(X_f, input_to_output_weights_f, bias_output_weights_f):
    output_pre_activation = np.dot(X_f, input_to_output_weights_f) + bias_output_weights_f
    return math_utils.sigmoid(output_pre_activation)
import numpy as np
import time

from utils import math_utils
from utils import displaying_nn_utils


def train_perceptron_nn(X_f, y_f, epochs_f, learning_rate_f):
    np.random.seed(42)

    input_neurons = X_f.shape[1]
    output_neurons = y_f.shape[1]

    input_to_output_weights = np.random.uniform(low=-1, high=1, size=(input_neurons, output_neurons))
    bias_output_weights = np.random.uniform(low=-1, high=1, size=(output_neurons, output_neurons))

    final_nn_output = np.array([])
    mse_values = []

    start_time = time.perf_counter()

    for epoch in range(epochs_f):
        # Forward propagation
        pre_activation_output = np.dot(X_f, input_to_output_weights) + bias_output_weights
        activated_output = math_utils.sigmoid(pre_activation_output)

        mse_values.append(math_utils.mean_squared_error(y_f, activated_output))

        final_nn_output = activated_output

        # Back propagation
        output_error = y_f - activated_output
        output_gradient = output_error * math_utils.sigmoid_derivative(activated_output)

        input_to_output_weights += learning_rate_f * np.dot(X_f.T, output_gradient)
        bias_output_weights += learning_rate_f * np.sum(output_gradient, axis=0, keepdims=True)

    end_time = time.perf_counter()
    training_time = end_time - start_time

    displaying_nn_utils.print_result_nn(final_nn_output, epochs_f, training_time)
    displaying_nn_utils.plot_mse(mse_values)

    return input_to_output_weights, bias_output_weights


def predict(X_f, weights, bias):
    final_input = np.dot(X_f, weights) + bias
    return math_utils.sigmoid(final_input)
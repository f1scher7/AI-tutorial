import numpy as np

from time import perf_counter
from utils.math.activation_funcs import activation_func
from utils.math.activation_funcs import activation_derivative_func
from utils.math.cost_funcs import *
from utils.visualisation import *


# Perceptrons are used for classification problems(mostly binary classification), typically use the sigmoid activation function
def train_perceptron_nn(x, y, epochs, learning_rate):
    np.random.seed(42)

    input_neurons = x.shape[1]
    output_neurons = y.shape[1]

    input_to_output_weights = np.random.uniform(low=-1, high=1, size=(input_neurons, output_neurons))
    bias_output_weights = np.random.uniform(low=-1, high=1, size=(1, output_neurons))

    final_nn_output = np.array([])
    mse_values = []

    start_time = perf_counter()

    for epoch in range(epochs):
        # Forward propagation
        output_pre_activation = np.dot(x, input_to_output_weights) + bias_output_weights
        output_activated = activation_func(output_pre_activation, 'sigmoid')

        mse_values.append(cost_func(y, output_activated, 'mse'))

        final_nn_output = output_activated

        # Back propagation
        output_error = y - output_activated
        output_gradient = output_error * activation_derivative_func(output_activated, 'sigmoid')

        input_to_output_weights += learning_rate * np.dot(x.T, output_gradient)
        bias_output_weights += learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

    end_time = perf_counter()
    training_time = end_time - start_time

    print_result_nn(final_nn_output, epochs, training_time)
    plot_mse(mse_values)

    return input_to_output_weights, bias_output_weights


def predict_perceptron_nn(X_f, input_to_output_weights_f, bias_output_weights_f):
    output_pre_activation = np.dot(X_f, input_to_output_weights_f) + bias_output_weights_f
    return activation_func(output_pre_activation, 'sigmoid')
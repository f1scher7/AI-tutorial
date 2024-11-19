import numpy as np

from time import perf_counter
from utils.math.activation_funcs import activation_func
from utils.math.activation_funcs import activation_derivative_func
from utils.math.cost_funcs import *

def train_single_layer_nn(X_f, y_f, hidden_neurons_f, epochs_f, learning_rate_f, plot_f):
    from utils.visualisation import plot_mse, plot_decision_boundary, print_result_nn

    np.random.seed(42)

    input_neurons = X_f.shape[1]
    output_neurons = y_f.shape[1]

    input_to_hidden_weights = np.random.uniform(low=-1, high=1, size=(input_neurons, hidden_neurons_f))
    hidden_to_output_weights = np.random.uniform(low=-1, high=1, size=(hidden_neurons_f, output_neurons))

    bias_hidden_weights = np.random.uniform(low=-1, high=1, size=(1, hidden_neurons_f))
    bias_output_weights = np.random.uniform(low=-1, high=1, size=(1, output_neurons))

    final_nn_output = np.array([])
    mse_values = []

    start_time = perf_counter()

    for epoch in range(epochs_f):
        # Forward propagation
        hidden_pre_activation = np.dot(X_f, input_to_hidden_weights) + bias_hidden_weights # Potted sum for hidden layer
        hidden_activated = activation_func(hidden_pre_activation, 'sigmoid') # Neurons activation for hidden layer

        output_pre_activation = np.dot(hidden_activated, hidden_to_output_weights) + bias_output_weights
        output_activated = activation_func(output_pre_activation, 'sigmoid')

        final_nn_output = output_activated
        mse_values.append(cost_func(y_f, output_activated, 'mse'))

        # Back propagation
        output_error = cost_derivative_func(y_f, output_activated, 'mse')
        # Output gradient - CORRECTIONS for each component
        # Output error * sensitivity for each component = how each component contributed to output error
        output_gradient = output_error * activation_derivative_func(output_activated, 'sigmoid')

        # Replacing output_error to hidden layer
        hidden_error = np.dot(output_gradient, hidden_to_output_weights.T)
        hidden_gradient = hidden_error * activation_derivative_func(hidden_activated, 'sigmoid')

        # Updating weights and biases
        hidden_to_output_weights += np.dot(hidden_activated.T, output_gradient) * learning_rate_f
        input_to_hidden_weights += np.dot(X_f.T, hidden_gradient) * learning_rate_f

        bias_output_weights += np.sum(output_gradient, axis=0, keepdims=True)
        bias_hidden_weights += np.sum(hidden_gradient, axis=0, keepdims=True)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse_values[:-1]}")

        if plot_f is not None:
            plot_decision_boundary(X_f, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights,bias_output_weights, epoch, plot_f)
            plot_f.pause(0.01)

    plot_f.show()

    end_time = perf_counter()
    training_time = end_time - start_time

    print_result_nn(final_nn_output, epochs_f, training_time)
    plot_mse(mse_values)

    return input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights


def predict_single_layer_nn(X_f, input_to_hidden_weights_f, hidden_to_output_weights_f, bias_hidden_weights_f, bias_output_weights_f):
    hidden_activated = activation_func(np.dot(X_f, input_to_hidden_weights_f) + bias_hidden_weights_f, 'sigmoid')
    return activation_func(np.dot(hidden_activated, hidden_to_output_weights_f) + bias_output_weights_f, 'sigmoid')

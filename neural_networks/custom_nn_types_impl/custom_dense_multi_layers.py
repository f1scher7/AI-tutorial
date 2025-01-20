import numpy as np

from time import perf_counter
from utils.math.activation_funcs import activation_func, activation_derivative_func
from utils.math.weights_initialization import weights_initialization_func
from utils.math.cost import cost_func, cost_derivative_func


class CustomDenseMultiLayers:

    def __init__(self, ):


def train_single_layer_nn(x, y, epochs, learning_rate, hidden_neurons, in_to_hid_init_name, hid_to_out_init_name, hid_act_func_name, out_act_func_name, plot):
    from utils.visualisation import plot_mse, plot_decision_boundary, print_result_nn

    np.random.seed(42)

    input_neurons = x.shape[1]
    output_neurons = y.shape[1]

    input_to_hidden_weights = weights_initialization_func((input_neurons, hidden_neurons), in_to_hid_init_name)
    hidden_to_output_weights = weights_initialization_func((hidden_neurons, output_neurons), hid_to_out_init_name)

    bias_hidden_weights = np.zeros((1, hidden_neurons))
    bias_output_weights = np.zeros((1, output_neurons))

    final_nn_output = np.array([])
    mse_values = []

    start_time = perf_counter()

    for epoch in range(epochs):
        # Forward propagation
        hidden_pre_activation = np.dot(x, input_to_hidden_weights) + bias_hidden_weights # Potted sum for hidden layer
        hidden_activated = activation_func(hidden_pre_activation, hid_act_func_name) # Neurons activation for hidden layer

        output_pre_activation = np.dot(hidden_activated, hidden_to_output_weights) + bias_output_weights
        output_activated = activation_func(output_pre_activation, out_act_func_name)

        final_nn_output = output_activated
        mse_values.append(cost_func(y, output_activated, 'mse'))

        # Back propagation
        output_error = y - output_activated
        # Output gradient - CORRECTIONS for each component
        # Output error * sensitivity for each component = how each component contributed to output error
        output_gradient = output_error * activation_derivative_func(output_activated, out_act_func_name)

        # Replacing output_error to hidden layer
        hidden_error = np.dot(output_gradient, hidden_to_output_weights.T)
        hidden_gradient = hidden_error * activation_derivative_func(hidden_activated, hid_act_func_name)

        # Updating weights and biases
        hidden_to_output_weights += np.dot(hidden_activated.T, output_gradient) * learning_rate
        input_to_hidden_weights += np.dot(x.T, hidden_gradient) * learning_rate

        bias_output_weights += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate
        bias_hidden_weights += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

        if plot is not None:
            plot_decision_boundary(x, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hid_act_func_name, out_act_func_name, epoch, plot)
            plot.pause(0.01)

    if plot is not None:
        plot.show()

    end_time = perf_counter()
    training_time = end_time - start_time

    print_result_nn(final_nn_output, epochs, training_time)
    plot_mse(mse_values)

    return input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights


def predict_single_layer_nn(x, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hid_act_func_name, out_act_func_name):
    hidden_activated = activation_func(np.dot(x, input_to_hidden_weights) + bias_hidden_weights, hid_act_func_name)
    return activation_func(np.dot(hidden_activated, hidden_to_output_weights) + bias_output_weights, out_act_func_name)

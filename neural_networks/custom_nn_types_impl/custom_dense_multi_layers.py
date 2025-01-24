import numpy as np
from time import perf_counter
from utils.math.activation_funcs import activation_func, activation_derivative_func
from utils.math.weights_initialization import weights_initialization_func
from utils.math.biases_initialization import biases_initialization_func
from utils.math.loss import training_loss_func, training_loss_derivative_func


class CustomDenseMultiLayerNN:

    def __init__(self, problem_name, input_data, target, hidden_neurons_list, activation_funcs_list,
                 weights_initialization_types, training_loss_function_name, epochs, learning_rate, momentum=0.9, reg_type=None, reg_lambda=0.1):

        self.problem_name = problem_name
        self.input_data = input_data
        self.target = target
        self.hidden_neurons_list = hidden_neurons_list
        self.activation_funcs_list = activation_funcs_list
        self.weights_initialization_types = weights_initialization_types
        self.training_loss_function_name = training_loss_function_name

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda

        self.batch_size = input_data.shape[0]
        self.sequence_len = input_data.shape[1]
        self.features_size = input_data.shape[2]

        self.num_hidden_layers = len(hidden_neurons_list)

        self.weights = []
        self.biases = []
        self.training_losses = []

        # adding weights for input -> hidden1 layer
        self.weights.append(weights_initialization_func((self.sequence_len, hidden_neurons_list[0]), self.weights_initialization_types[0]))
        self.biases.append(biases_initialization_func(self.hidden_neurons_list[0], self.activation_funcs_list[0]))

        if self.num_hidden_layers > 1:
            for layer_idx in range(1, self.num_hidden_layers):
                self.weights.append(weights_initialization_func((self.hidden_neurons_list[layer_idx - 1], self.hidden_neurons_list[layer_idx]), weights_initialization_types[layer_idx]))
                self.biases.append(np.zeros(self.hidden_neurons_list[layer_idx]))

        # adding weights and biases for hidden_last -> output layer
        self.weights.append(weights_initialization_func((hidden_neurons_list[-1], self.target.shape[1]), weights_initialization_types[-1]))
        self.biases.append(biases_initialization_func(self.target.shape[1], self.activation_funcs_list[-1]))

        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]


    def forward_propagation(self):
        activated_hidden_layers = []
        activated_outputs = []

        for i in range(self.batch_size):
            activated_hidden_layers_for_batch = []

            prev_activated_layer = self.input_data[i]

            for layer_idx in range(self.num_hidden_layers):
                raw_layer = np.dot(prev_activated_layer, self.weights[layer_idx]) + self.biases[layer_idx]
                activated_layer = activation_func(raw_layer, self.activation_funcs_list[layer_idx])

                activated_hidden_layers_for_batch.append(activated_layer)
                prev_activated_layer = activated_layer

            raw_output = np.dot(prev_activated_layer, self.weights[-1]) + self.biases[-1]
            activated_output = activation_func(raw_output, self.activation_funcs_list[-1])

            activated_hidden_layers.append(activated_hidden_layers_for_batch)
            activated_outputs.append(activated_output)

        return activated_hidden_layers, activated_outputs


    def back_propagation(self, activated_hidden_layers, activated_outputs):
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        for i in range(self.batch_size):
            y_true = self.target[i]
            y_pred = activated_outputs[i]

            self.training_losses.append(training_loss_func(y_true=y_true, y_pred=y_pred, training_loss_func_name=self.training_loss_function_name))
            error = training_loss_derivative_func(y_true=y_true, y_pred=y_pred, training_loss_func_name=self.training_loss_function_name)

            delta = error * activation_derivative_func(y_pred, self.activation_funcs_list[-1])

            d_weights[-1] += np.outer(activated_hidden_layers[-1], delta)
            d_biases[-1] += delta

            for layer_idx in range(self.num_hidden_layers - 1, -1, -1):
                error = np.dot(delta, self.weights[layer_idx + 1].T)
                delta = error * activation_derivative_func(activated_hidden_layers[layer_idx], self.activation_funcs_list[layer_idx])

                prev_activation_layer = self.input_data[i] if layer_idx == 0 else activated_hidden_layers[layer_idx - 1]

                d_weights[layer_idx] += np.outer(prev_activation_layer, delta)
                d_biases[layer_idx] += delta

        d_weights = [dw / self.batch_size for dw in d_weights]
        d_biases = [db / self.batch_size for db in d_biases]

        for layer_idx in range(len(self.weights)):
            self.velocity_weights[layer_idx] = (
                self.momentum * self.velocity_weights[layer_idx] - self.learning_rate * d_weights[layer_idx]
            )

            self.velocity_biases[layer_idx] = (
                self.momentum * self.velocity_biases[layer_idx] - self.learning_rate * d_biases[layer_idx]
            )

            # add additional "speed" for weights
            self.weights[layer_idx] += self.velocity_weights[layer_idx]
            self.biases[layer_idx] += self.velocity_biases[layer_idx]







# def train_single_layer_nn(x, y, epochs, learning_rate, hidden_neurons, in_to_hid_init_name, hid_to_out_init_name, hid_act_func_name, out_act_func_name, plot):
#     from utils.visualisation import plot_mse, plot_decision_boundary, print_result_nn
#
#     np.random.seed(42)
#
#     input_neurons = x.shape[1]
#     output_neurons = y.shape[1]
#
#     input_to_hidden_weights = weights_initialization_func((input_neurons, hidden_neurons), in_to_hid_init_name)
#     hidden_to_output_weights = weights_initialization_func((hidden_neurons, output_neurons), hid_to_out_init_name)
#
#     bias_hidden_weights = np.zeros((1, hidden_neurons))
#     bias_output_weights = np.zeros((1, output_neurons))
#
#     final_nn_output = np.array([])
#     mse_values = []
#
#     start_time = perf_counter()
#
#     for epoch in range(epochs):
#         # Forward propagation
#         hidden_pre_activation = np.dot(x, input_to_hidden_weights) + bias_hidden_weights # Potted sum for hidden layer
#         hidden_activated = activation_func(hidden_pre_activation, hid_act_func_name) # Neurons activation for hidden layer
#
#         output_pre_activation = np.dot(hidden_activated, hidden_to_output_weights) + bias_output_weights
#         output_activated = activation_func(output_pre_activation, out_act_func_name)
#
#         final_nn_output = output_activated
#         mse_values.append(cost_func(y, output_activated, 'mse'))
#
#         # Back propagation
#         output_error = y - output_activated
#         # Output gradient - CORRECTIONS for each component
#         # Output error * sensitivity for each component = how each component contributed to output error
#         output_gradient = output_error * activation_derivative_func(output_activated, out_act_func_name)
#
#         # Replacing output_error to hidden layer
#         hidden_error = np.dot(output_gradient, hidden_to_output_weights.T)
#         hidden_gradient = hidden_error * activation_derivative_func(hidden_activated, hid_act_func_name)
#
#         # Updating weights and biases
#         hidden_to_output_weights += np.dot(hidden_activated.T, output_gradient) * learning_rate
#         input_to_hidden_weights += np.dot(x.T, hidden_gradient) * learning_rate
#
#         bias_output_weights += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate
#         bias_hidden_weights += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate
#
#         if plot is not None:
#             plot_decision_boundary(x, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hid_act_func_name, out_act_func_name, epoch, plot)
#             plot.pause(0.01)
#
#     if plot is not None:
#         plot.show()
#
#     end_time = perf_counter()
#     training_time = end_time - start_time
#
#     print_result_nn(final_nn_output, epochs, training_time)
#     plot_mse(mse_values)
#
#     return input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights
#
#
# def predict_single_layer_nn(x, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hid_act_func_name, out_act_func_name):
#     hidden_activated = activation_func(np.dot(x, input_to_hidden_weights) + bias_hidden_weights, hid_act_func_name)
#     return activation_func(np.dot(hidden_activated, hidden_to_output_weights) + bias_output_weights, out_act_func_name)

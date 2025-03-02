import numpy as np
from time import perf_counter
from utils.math.activation_funcs import activation_func, activation_derivative_func
from utils.math.weights_initialization import weights_initialization_func
from utils.math.biases_initialization import biases_initialization_func
from utils.math.loss import training_loss_func, training_loss_derivative_func
from utils.visualization import print_training_logs_nn, plot_training_losses
from utils.utils import save_nn_model, load_saved_nn_model


class CustomDenseMultiLayerNN:

    def __init__(self, problem_name, input_data_norm, target_norm,
                 data_norm_func_name, input_data_norm_params, target_norm_params,
                 hidden_neurons_list, activ_funcs_list, weights_init_types_list,
                 training_loss_func_name, epochs, learning_rate, momentum=0.9, reg_type=None, reg_lambda=0.1, is_train=True):

        self.problem_name = problem_name
        self.input_data_norm = input_data_norm
        self.target_norm = target_norm

        self.data_norm_func_name = data_norm_func_name
        self.input_data_norm_params = input_data_norm_params
        self.target_norm_params = target_norm_params

        self.hidden_neurons_list = hidden_neurons_list
        self.activ_funcs_list = activ_funcs_list
        self.weights_init_types_list = weights_init_types_list

        self.training_loss_func_name = training_loss_func_name

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.is_train = is_train

        self.batch_size = input_data_norm.shape[0] if input_data_norm is not None else 1
        self.sequence_len = input_data_norm.shape[1] if input_data_norm is not None else None
        self.features_size = input_data_norm.shape[2] if input_data_norm is not None else None

        self.num_hidden_layers = len(self.hidden_neurons_list)

        self.weights = []
        self.biases = []
        self.training_losses = []
        self.velocity_weights = []

        self.is_weights_and_biases_init = False

        if self.is_train:
            # adding weights for input -> hidden1 layer
            if self.features_size is not None:
                self.init_weights_and_biases()
                self.is_weights_and_biases_init = True


    def inference(self, input_data):
        self.input_data_norm = input_data

        self.batch_size = 1
        self.sequence_len = self.input_data_norm.shape[0]
        self.features_size = self.input_data_norm.shape[1]

        _, activated_outputs = self.forward_propagation()

        return activated_outputs


    def train(self, is_save):
        start_time = perf_counter()

        last_output =  None

        for epoch in range(self.epochs):
            activated_hidden_layers, activated_outputs = self.forward_propagation()
            self.back_propagation(activated_hidden_layers, activated_outputs)

            if epoch % 100 == 0:
                print_training_logs_nn(nn_name='DenseMultiLayerNN', epoch=epoch + 1, epochs=self.epochs, training_loss=self.training_losses[-1], training_loss_function_name=self.training_loss_func_name)

            if epoch + 1 == self.epochs:
                last_output = activated_outputs

        end_time = perf_counter()
        training_time = end_time - start_time

        print(f'Last output: {last_output}')
        print(f'Training time: {training_time:.2f} secs')

        if is_save: self.save()

        plot_training_losses(problem_name=self.problem_name, training_loss_function_name=self.training_loss_func_name, training_losses=self.training_losses)


    def forward_propagation(self):
        if self.is_train and not self.is_weights_and_biases_init and self.features_size is not None:
            self.init_weights_and_biases()
            self.is_weights_and_biases_init = True

        activated_hidden_layers = []
        activated_outputs = []

        for i in range(self.batch_size):
            activated_hidden_layers_for_batch = []

            prev_activated_layer = self.input_data_norm[i]

            for layer_idx in range(self.num_hidden_layers):
                raw_layer = np.dot(prev_activated_layer, self.weights[layer_idx]) + self.biases[layer_idx]
                activated_layer = activation_func(raw_layer, self.activ_funcs_list[layer_idx])

                activated_hidden_layers_for_batch.append(activated_layer)
                prev_activated_layer = activated_layer

            raw_output = np.dot(prev_activated_layer, self.weights[-1]) + self.biases[-1]
            activated_output = activation_func(raw_output, self.activ_funcs_list[-1])

            activated_hidden_layers.append(activated_hidden_layers_for_batch)
            activated_outputs.append(activated_output)

        return activated_hidden_layers, activated_outputs


    def back_propagation(self, activated_hidden_layers, activated_outputs):
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        all_input_deltas = []

        batch_loss = 0

        is_prev_layer_input = False

        for i in range(self.batch_size):
            batch_loss += training_loss_func(y_true=self.target_norm[i], y_pred=activated_outputs[i], batch_size=self.batch_size, training_loss_func_name=self.training_loss_func_name)
            error = training_loss_derivative_func(y_true=self.target_norm[i], y_pred=activated_outputs[i], batch_size=self.batch_size, training_loss_func_name=self.training_loss_func_name)

            if self.activ_funcs_list[-1] == 'softmax' and self.training_loss_func_name == 'categorical_cross_entropy':
                delta = activated_outputs[i] - self.target_norm[i]
            else:
                delta = error * activation_derivative_func(activated_outputs[i], self.activ_funcs_list[-1])

            if activated_hidden_layers[i][-1].ndim == 1 and delta.ndim == 1:
                activated_hidden_layers[i][-1] = activated_hidden_layers[i][-1].reshape(1, -1)
                delta = delta.reshape(1, -1)

            d_weights[-1] += np.dot(activated_hidden_layers[i][-1].T, delta)
            d_biases[-1] += np.sum(delta, axis=0)

            prev_delta = delta

            for layer_idx in range(self.num_hidden_layers - 1, -1, -1):
                if layer_idx == self.num_hidden_layers - 1:
                    error = np.dot(prev_delta, self.weights[-1].T)
                else:
                    error = np.dot(prev_delta, self.weights[layer_idx + 1].T)

                delta = error * activation_derivative_func(activated_hidden_layers[i][layer_idx], self.activ_funcs_list[layer_idx])

                if layer_idx == 0:
                    prev_layer = self.input_data_norm[i]
                    is_prev_layer_input = True
                else:
                    prev_layer = activated_hidden_layers[i][layer_idx - 1]

                if prev_layer.ndim == 1:
                    prev_layer = prev_layer.reshape(1, -1)

                d_weights[layer_idx] += np.dot(prev_layer.T, delta)
                d_biases[layer_idx] += np.sum(delta, axis=0)

                prev_delta = delta

                if is_prev_layer_input:
                    all_input_deltas.append(np.dot(delta, self.weights[0].T))

            is_prev_layer_input = False

        self.training_losses.append(batch_loss / self.batch_size)

        # Average gradients
        d_weights = [dw / self.batch_size for dw in d_weights]
        d_biases = [db / self.batch_size for db in d_biases]

        for layer_idx in range(self.num_hidden_layers + 1):
            self.velocity_weights[layer_idx] = (
                self.momentum * self.velocity_weights[layer_idx] - self.learning_rate * d_weights[layer_idx]
            )

            # Add additional "speed" for weights
            self.weights[layer_idx] += self.velocity_weights[layer_idx]

            self.biases[layer_idx] -= self.learning_rate * d_biases[layer_idx]

        return self.weights, self.biases, np.array(all_input_deltas).reshape(self.batch_size, -1)


    def init_weights_and_biases(self):
        self.weights.append(weights_initialization_func((self.features_size, self.hidden_neurons_list[0]), self.weights_init_types_list[0]))
        self.biases.append(biases_initialization_func(self.hidden_neurons_list[0], self.activ_funcs_list[0]))

        if self.num_hidden_layers > 1:
            for layer_idx in range(1, self.num_hidden_layers):
                self.weights.append(weights_initialization_func(
                    (self.hidden_neurons_list[layer_idx - 1], self.hidden_neurons_list[layer_idx]), self.weights_init_types_list[layer_idx]))
                self.biases.append(np.zeros(self.hidden_neurons_list[layer_idx]))

        # adding weights and biases for hidden_last -> output layer
        self.weights.append(weights_initialization_func((self.hidden_neurons_list[-1], self.target_norm.shape[-1]), self.weights_init_types_list[-1]))
        self.biases.append(biases_initialization_func(self.target_norm.shape[-1], self.activ_funcs_list[-1]))

        self.velocity_weights = [np.zeros_like(w) for w in self.weights]


    def save(self):
        model_info = self.data_to_save()

        file_name = f'{self.problem_name}_CustomDenseMultiLayerNN'

        save_nn_model(file_name, model_info)


    def data_to_save(self):
        return {
            "problem_name": self.problem_name,

            "weights": self.weights,
            "biases": self.biases,

            "data_norm_func_name": self.data_norm_func_name,
            "input_data_norm_params": self.input_data_norm_params,
            "target_norm_params": self.target_norm_params,

            "hidden_neurons_list": self.hidden_neurons_list,
            "activ_funcs_list": self.activ_funcs_list,
            "weights_init_types_list": self.weights_init_types_list,

            "batch_size": self.batch_size,
            "sequence_len": self.sequence_len,
            "features_size": self.features_size,

            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,

            "reg_type": self.reg_type,
            "reg_lambda": self.reg_lambda,

            "training_loss_func_name": self.training_loss_func_name,
            "training_losses": self.training_losses,
        }


def load_custom_dense_multilayer_nn(file_name):
    model_info = load_saved_nn_model(file_name=file_name)

    dense_multilayer_nn = CustomDenseMultiLayerNN(
        problem_name=model_info['problem_name'], input_data_norm=None, target_norm=None,
        data_norm_func_name=model_info['data_norm_func_name'], input_data_norm_params=model_info['input_data_norm_params'], target_norm_params=model_info['target_norm_params'],
        hidden_neurons_list=model_info['hidden_neurons_list'], activ_funcs_list=model_info['activ_funcs_list'],
        weights_init_types_list=model_info['weights_init_types_list'], training_loss_func_name=model_info['training_loss_func_name'],
        epochs=model_info['epochs'], learning_rate=model_info['learning_rate'], momentum=model_info['momentum'], reg_type=model_info['reg_type'], reg_lambda=model_info['reg_lambda'], is_train=False
    )

    dense_multilayer_nn.weights = model_info['weights']
    dense_multilayer_nn.biases = model_info['biases']

    return dense_multilayer_nn


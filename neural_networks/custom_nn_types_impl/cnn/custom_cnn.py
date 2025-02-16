import numpy as np
from time import perf_counter
from neural_networks.custom_nn_types_impl.cnn.convolutional_layer import ConvolutionLayer
from neural_networks.custom_nn_types_impl.cnn.max_pooling_layer import MaxPoolingLayer
from neural_networks.custom_nn_types_impl.cnn.flatten_layer import FlattenLayer
from neural_networks.custom_nn_types_impl.custom_dense_multi_layers import CustomDenseMultiLayerNN
from enums import WeightsInitFuncName, ActivFuncName
from utils.visualization import print_training_logs_nn, plot_training_losses
from utils.utils import save_nn_model, load_saved_nn_model


class CustomCnn:

    def __init__(self, problem_name, input_data, target, filters_num, filter_size, pool_size,
                 hidden_neurons_list_for_dense, activ_func_name_list_for_dense, weights_init_types_list_for_dense, training_loss_func_name,
                 filter_init_func_name=WeightsInitFuncName.HE.value, feature_map_activ_name=ActivFuncName.RELU.value,
                 stride_for_conv_layer=1, stride_for_max_pooling_layer=1, padding=0, is_save=True):

        self.conv_layer = ConvolutionLayer(input_data=input_data, filters_num=filters_num, filter_size=filter_size, filter_init_func_name=filter_init_func_name,
                                           stride=stride_for_conv_layer, padding=padding, feature_map_activ_func=feature_map_activ_name)

        self.max_pooling_layer = MaxPoolingLayer(pool_size=pool_size, stride=stride_for_max_pooling_layer)

        self.flatten_layer = FlattenLayer()

        self.custom_dense_multi_layers = CustomDenseMultiLayerNN(problem_name=problem_name, input_data_norm=None, target_norm=target,
                                                                 data_norm_func_name=None, input_data_norm_params=None, target_norm_params=None,
                                                                 hidden_neurons_list=hidden_neurons_list_for_dense, activ_funcs_list=activ_func_name_list_for_dense,
                                                                 weights_init_types_list=weights_init_types_list_for_dense,
                                                                 training_loss_func_name=training_loss_func_name, epochs=None, learning_rate=None, is_train=True)
        self.problem_name = problem_name
        self.epochs = None
        self.is_save = is_save


    def train(self, epochs, learning_rate_conv, learning_rate_dense):
        self.epochs = epochs
        self.conv_layer.learning_rate = learning_rate_conv
        self.custom_dense_multi_layers.learning_rate = learning_rate_dense

        last_outputs = None

        start_time = perf_counter()

        for epoch in range(self.epochs):
            activated_feature_maps = self.conv_layer.forward_propagation()
            pooling_results = self.max_pooling_layer.forward_propagation(feature_maps=activated_feature_maps)
            flatten_data = self.flatten_layer.forward_propagation(input_data=pooling_results)

            self.prepare_custom_dense_multi_layers_for_forward_propagation(data=flatten_data)
            activated_dense_hidden_layers, activated_dense_outputs = self.custom_dense_multi_layers.forward_propagation()

            _, _, all_deltas = self.custom_dense_multi_layers.back_propagation(activated_hidden_layers=activated_dense_hidden_layers, activated_outputs=activated_dense_outputs)
            unflatten_data = self.flatten_layer.back_propagation(output_data=all_deltas)
            d_feature_maps = self.max_pooling_layer.back_propagation(d_outputs=unflatten_data)
            self.conv_layer.back_propagation(d_feature_maps=d_feature_maps)

            if epoch % 2 == 0:
                print_training_logs_nn(nn_name='DenseMultiLayerNN', epoch=epoch + 1, epochs=self.epochs, training_loss=self.custom_dense_multi_layers.training_losses[-1], training_loss_function_name=self.custom_dense_multi_layers.training_loss_func_name)

            if epoch + 1 == self.epochs:
                last_outputs = activated_dense_outputs

        end_time = perf_counter()
        training_time = end_time - start_time

        print(f'Last output: {last_outputs}')
        print(f'Training time: {training_time:.2f} secs')

        if self.is_save: self.save()

        plot_training_losses(problem_name=self.problem_name, training_loss_function_name=self.custom_dense_multi_layers.training_loss_func_name, training_losses=self.custom_dense_multi_layers.training_losses)


    def prepare_custom_dense_multi_layers_for_forward_propagation(self, data):
        self.custom_dense_multi_layers.input_data_norm = data
        self.custom_dense_multi_layers.batch_size = data.shape[0]
        self.custom_dense_multi_layers.sequence_len = 1
        self.custom_dense_multi_layers.features_size = data.shape[1]


    def save(self):
        model_info = self.data_to_save()

        file_name = f'{self.problem_name}_CustomCnn'

        save_nn_model(file_name, model_info)


    def data_to_save(self):
        conv_layer_data = self.conv_layer.data_to_save()
        max_pooling_data = self.max_pooling_layer.data_to_save()
        flatten_data = self.flatten_layer.data_to_save()
        custom_dense_multi_layers_data = self.custom_dense_multi_layers.data_to_save()

        return {
            "conv_layer_data": conv_layer_data,
            "max_pooling_data": max_pooling_data,
            "flatten_data": flatten_data,
            "custom_dense_multi_layers_data": custom_dense_multi_layers_data,
        }


def load_custom_cnn(file_name):
    model_info = load_saved_nn_model(file_name)

    return model_info
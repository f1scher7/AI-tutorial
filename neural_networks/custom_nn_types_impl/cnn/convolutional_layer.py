import numpy as np
from utils.math.weights_initialization import cnn_weights_initialization_func
from utils.math.biases_initialization import biases_initialization_func
from utils.math.activation_funcs import activation_func, activation_derivative_func
from enums import WeightsInitFuncName, ActivFuncName


class ConvolutionLayer:

    def __init__(self, input_data, filters_num, filter_size, filter_init_func_name=WeightsInitFuncName.HE.value,
                 stride=1, padding=0, feature_map_activ_func=ActivFuncName.RELU.value, learning_rate=0.01):
        """
        input_data: (batch_size, img_height, img_weight, channels)
        filter_size: (filter_width, filter_height)
        """
        self.original_input_data = input_data
        self.input_data = input_data
        self.filters_num = filters_num
        self.filter_size = filter_size
        self.filter_init_func_name = filter_init_func_name
        self.stride = stride
        self.padding = padding
        self.feature_map_activ_func = feature_map_activ_func
        self.learning_rate = learning_rate

        self.batch_size = self.input_data.shape[0]
        self.input_height = self.input_data.shape[1]
        self.input_width = self.input_data.shape[2]
        self.channels = self.input_data.shape[3]

        self.filter_width = self.filter_size[0]
        self.filter_height = self.filter_size[1]

        self.filters = cnn_weights_initialization_func(shape=(self.filters_num, self.filter_height, self.filter_width, self.channels), func_name=self.filter_init_func_name)
        self.biases = biases_initialization_func(bias_len=self.filters_num, func_name=self.feature_map_activ_func)

        self.feature_map_width = (self.input_width - self.filter_width + 2 * self.padding) // self.stride + 1
        self.feature_map_height = (self.input_height - self.filter_height + 2 * self.padding) // self.stride + 1

        self.pre_activated_feature_maps = None

        self.apply_padding()


    def forward_propagation(self):
        feature_maps = np.zeros((self.batch_size, self.filters_num, self.feature_map_height, self.feature_map_width))
        activated_feature_maps = np.zeros_like(feature_maps)
        self.pre_activated_feature_maps = np.zeros_like(feature_maps)

        for batch_idx in range(self.batch_size):
            img = self.input_data[batch_idx]

            for flt_idx in range(self.filters_num):
                feature_maps[batch_idx, flt_idx] = self.apply_convolution(img, self.filters[flt_idx], self.feature_map_height, self.feature_map_width)
                feature_maps[batch_idx, flt_idx] += self.biases[flt_idx]

                self.pre_activated_feature_maps[batch_idx, flt_idx] = feature_maps[batch_idx, flt_idx]

                activated_feature_maps[batch_idx, flt_idx] = activation_func(feature_maps[batch_idx, flt_idx], self.feature_map_activ_func)

        return activated_feature_maps


    def back_propagation(self, d_feature_maps):
        d_biases = np.sum(d_feature_maps, axis=(0, 2, 3))
        d_filters = np.zeros_like(self.filters)
        d_input_data = np.zeros_like(self.input_data) # self.input_data is the self.original_input_data with padding

        for batch_idx in range(self.batch_size):
            d_feature_maps[batch_idx] *= activation_derivative_func(self.pre_activated_feature_maps[batch_idx], self.feature_map_activ_func)
            for flt_idx in range(self.filters_num):

                for i in range(self.feature_map_height):
                    for j in range(self.feature_map_width):
                        start_i = i * self.stride
                        start_j = j * self.stride

                        img_slice = self.input_data[batch_idx, start_i:start_i + self.filter_height, start_j:start_j + self.filter_width]

                        d_filters[flt_idx] += img_slice * d_feature_maps[batch_idx, flt_idx, i, j]
                        d_input_data[batch_idx, start_i:start_i + self.filter_height, start_j:start_j + self.filter_width] += self.filters[flt_idx] * d_feature_maps[batch_idx, flt_idx, i, j]

        self.filters -= self.learning_rate * d_filters
        self.biases -= self.learning_rate * d_biases

        # return d_input_data # we should return d_input_data for the prev layer


    def apply_padding(self):
        # applying padding for original img
        if self.padding > 0:
            self.input_data = np.pad(self.input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)


    def apply_convolution(self, img, flt, feature_map_height, feature_map_width):
        feature_map = np.zeros((feature_map_height, feature_map_width))

        for i in range(feature_map_height):
            for j in range(feature_map_width):
                start_i = i * self.stride
                start_j = j * self.stride

                img_slice = img[start_i:start_i + flt.shape[0], start_j:start_j + flt.shape[1]]
                feature_map[i, j] = np.sum(np.multiply(img_slice, flt), axis=(0, 1, 2))

        return feature_map


    def data_to_save(self):
        return {
            "filters": self.filters,
            "biases": self.biases,

            "filter_init_func_name": self.filter_init_func_name,
            "filters_num": self.filters_num,
            "filter_size": self.filter_size,
            "filter_height": self.filter_height,
            "filter_width": self.filter_width,

            "pre_activated_feature_maps": self.pre_activated_feature_maps,
            "feature_map_activ_func": self.feature_map_activ_func,
            "feature_map_width": self.feature_map_width,
            "feature_map_height": self.feature_map_height,

            "padding": self.padding,
            "stride": self.stride,

            "batch_size": self.batch_size,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "channels": self.channels,

            "learning_rate": self.learning_rate,
        }
import numpy as np
from utils.math.weights_initialization import cnn_weights_initialization_func
from utils.math.biases_initialization import biases_initialization_func


class ConvolutionLayer:

    def __init__(self, input_shape, filters_num, filter_size, stride=1, padding=0):
        """
        input_shape: (width, height, channels)
        filter_size: (filter_width, filter_height)
        """
        self.input_shape = input_shape
        self.filters_num = filters_num
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.input_width = self.input_shape[0]
        self.input_height = self.input_shape[1]
        self.channels = input_shape[2]

        self.filter_width = self.filter_size[0]
        self.filter_height = self.filter_size[1]

        self.filters = cnn_weights_initialization_func(shape=(self.filters_num, self.filter_height, self.filter_width, self.channels), func_name='he')
        self.biases = biases_initialization_func(bias_len=self.filters_num, func_name='relu')

        self.feature_map_width = (self.input_width - self.filter_width + 2 * self.padding) // self.stride + 1
        self.feature_map_height = (self.input_height - self.filter_height + 2 * self.padding) // self.stride + 1


    def forward_propagation(self, input_data):
        """
        input_data: (batch_size, img_height, img_weight, channels)
        """
        batch_size = input_data.shape[0]

        # applying padding for original img
        if self.padding > 0:
            input_data = np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)

        features_maps = np.zeros((batch_size, self.filters_num, self.feature_map_height, self.feature_map_width))

        for batch in range(batch_size):
            img = input_data[batch]

            for flt_idx in range(self.filters_num):
                features_maps[batch, flt_idx] = self.apply_convolution(img, self.filters[flt_idx], self.feature_map_height, self.feature_map_width)

        return features_maps


    def apply_convolution(self, img, flt, feature_map_height, feature_map_width):
        feature_map = np.zeros((feature_map_height, feature_map_width))

        for i in range(feature_map_height):
            for j in range(feature_map_width):
                start_i = i * self.stride
                start_j = j * self.stride

                img_slice = img[start_i:start_i + flt.shape[0], start_j:start_j + flt.shape[1]]
                feature_map[i, j] = np.sum(img_slice * flt)

        return feature_map

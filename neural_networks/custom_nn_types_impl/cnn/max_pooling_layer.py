import numpy as np


class MaxPoolingLayer:

    def __init__(self, pool_size, stride):
        self.pool_height = pool_size[0]
        self.pool_width = pool_size[1]
        self.stride = stride


    def forward_propagation(self, feature_maps):
        """
        feature_maps (batch_size, num_feature_maps, feature_map_height, feature_map_width)
        """
        batch_size = feature_maps.shape[0]
        feature_maps_num = feature_maps.shape[1]

        pooling_result_height = (feature_maps.shape[2] - self.pool_height) // self.stride + 1
        pooling_result_width = (feature_maps.shape[3] - self.pool_width) // self.stride + 1
        pooling_results = np.zeros((batch_size, feature_maps_num, pooling_result_height, pooling_result_width))

        for batch_idx in range(batch_size):
            for map_idx in range(feature_maps_num):
                feature_map = feature_maps[batch_idx, map_idx]

                pooling_results[batch_idx, map_idx] = self.apply_max_pooling(feature_map, pooling_result_height, pooling_result_width)

        return pooling_results


    def apply_max_pooling(self, feature_map, pooling_result_height, pooling_result_width):
        pooling_result = np.zeros((pooling_result_height, pooling_result_width))

        for i in range(pooling_result_height):
            for j in range(pooling_result_width):
                start_i = i * self.stride
                start_j = j * self.stride

                feature_map_slice = feature_map[start_i:start_i + self.pool_height, start_j:start_j + self.pool_width]
                pooling_result[i, j] = np.max(feature_map_slice)

        return pooling_result

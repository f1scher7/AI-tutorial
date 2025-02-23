import numpy as np


class MaxPoolingLayer:

    def __init__(self, pool_size, stride):
        self.pool_height = pool_size[0]
        self.pool_width = pool_size[1]
        self.stride = stride
        self.masks = None # we need the self.masks to store max value position in the feature_map


    def forward_propagation(self, feature_maps):
        """
        feature_maps (batch_size, num_feature_maps, feature_map_height, feature_map_width)
        """
        batch_size = feature_maps.shape[0]
        feature_maps_num = feature_maps.shape[1]

        pooling_result_height = (feature_maps.shape[2] - self.pool_height) // self.stride + 1
        pooling_result_width = (feature_maps.shape[3] - self.pool_width) // self.stride + 1
        pooling_results = np.zeros((batch_size, feature_maps_num, pooling_result_height, pooling_result_width))

        self.masks = np.zeros_like(feature_maps, dtype=bool)

        for batch_idx in range(batch_size):
            for map_idx in range(feature_maps_num):
                feature_map = feature_maps[batch_idx, map_idx]

                pooling_results[batch_idx, map_idx], self.masks[batch_idx, map_idx] = self.apply_max_pooling(feature_map, pooling_result_height, pooling_result_width)

        return pooling_results


    def back_propagation(self, d_outputs):
        """
        d_outputs: are gradients from the dense nn which were reshaped to pooling_results_size in the back propagation of flatten layer
        """
        batch_size, feature_maps_num, pooling_result_height, pooling_result_width = d_outputs.shape
        d_feature_maps = np.zeros_like(self.masks, dtype=np.float64)

        for batch_idx in range(batch_size):
            for map_idx in range(feature_maps_num):
                for i in range(pooling_result_height):
                    for j in range(pooling_result_width):
                        start_i = i * self.stride
                        start_j = j * self.stride

                        current_mask = self.masks[batch_idx, map_idx, start_i:start_i + self.pool_height, start_j:start_j + self.pool_width].astype(np.float64)

                        d_feature_maps[batch_idx, map_idx, start_i:start_i + self.pool_height, start_j:start_j + self.pool_width] += current_mask * d_outputs[batch_idx, map_idx, i, j]

        return d_feature_maps


    def apply_max_pooling(self, feature_map, pooling_result_height, pooling_result_width):
        pooling_result = np.zeros((pooling_result_height, pooling_result_width))
        mask = np.zeros(feature_map.shape)

        for i in range(pooling_result_height):
            for j in range(pooling_result_width):
                start_i = i * self.stride
                start_j = j * self.stride

                feature_map_slice = feature_map[start_i:start_i + self.pool_height, start_j:start_j + self.pool_width]

                max_value = np.max(feature_map_slice)

                max_value_positions = np.argwhere(feature_map_slice == max_value) # we find the positions of the max value in feature_map_slice

                rand_chosen_max_value_pos = max_value_positions[np.random.choice(len(max_value_positions))]

                mask[start_i + rand_chosen_max_value_pos[0], start_j + rand_chosen_max_value_pos[1]] = 1

                pooling_result[i, j] = max_value

        return pooling_result, mask


    def data_to_save(self):
        return {
            "pool_height": self.pool_height,
            "pool_width": self.pool_width,
            "stride": self.stride,
            "masks": self.masks,
        }

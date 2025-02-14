import numpy as np


class FlattenLayer:

    def __init__(self):
        self.input_shape = None


    def forward_propagation(self, input_data):
        self.input_shape = input_data.shape

        batch_size = input_data.shape[0]

        return input_data.reshape(batch_size, -1)


    def back_propagation(self, output_data):
        return output_data.reshape(self.input_shape)


    def data_to_save(self):
        return {
            "input_shape": self.input_shape
        }
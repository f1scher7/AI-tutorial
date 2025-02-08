from enum import Enum


class ActivFuncName(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    SIGMOID_RAW = 'sigmoid_raw'
    TANH = 'tanh'
    TANH_RAW = 'tanh_raw'
    SOFTMAX = 'softmax'


class LossFuncName(Enum):
    MSE = 'mse'
    BINARY_CROSS_ENTROPY = 'binary_cross_entropy'
    CATEGORICAL_CROSS_ENTROPY = 'categorical_cross_entropy'


class WeightsInitFuncName(Enum):
    RANDOM = 'random'
    NORMAL = 'normal'
    HE = 'he'
    XAVIER = 'xavier'
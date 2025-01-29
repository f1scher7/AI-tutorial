import numpy as np

np.random.seed(42)


def weights_initialization_func(shape, func_name):
    return weights_initialization_funcs[func_name](shape)


weights_initialization_funcs = {
    'random': lambda shape: np.random.uniform(low=-1, high=1, size=shape),
    'normal': lambda shape: np.random.randn(*shape),
    'he': lambda shape: np.random.normal(loc=0, scale=np.sqrt(2. / shape[0]), size=shape),
    'xavier': lambda shape: np.random.uniform(low=-np.sqrt(2. / (shape[0] + shape[1])), high=np.sqrt(2. / (shape[0] + shape[1])), size=shape)
}


def cnn_weights_initialization_func(shape, func_name):
    """
    shape: (filter_nums, filter_height, filter_width, channels)
    """
    if func_name == 'random' or func_name == 'normal':
        return weights_initialization_func(shape=shape, func_name=func_name)
    else:
        filter_area = shape[1] * shape[2]
        fan_in = shape[3] * filter_area

        if func_name == 'he':
            return np.random.normal(loc=0, scale=np.sqrt(2. / fan_in), size=shape)
        elif func_name == 'xavier':
            fan_out = shape[0] * filter_area
            return np.random.uniform(low=-np.sqrt(2. / (fan_in + fan_out)), high=np.sqrt(2. / (fan_in + fan_out)), size=shape)

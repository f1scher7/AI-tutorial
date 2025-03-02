import numpy as np
from data_processing import load_processed_dataset
from enums import ActivFuncName, WeightsInitFuncName, LossFuncName
from neural_networks.custom_nn_types_impl.cnn.custom_cnn import CustomCnn, load_custom_cnn

problem_name = 'DIGITS-CLASSIFICATION-problem'
input_data, target = load_processed_dataset()

filters_num = 8
filter_size = (3, 3)
pool_size = (2, 2)

filter_init_func_name = WeightsInitFuncName.HE.value
feature_map_activ_name = ActivFuncName.RELU.value

stride_for_conv_layer = 1
stride_for_max_pooling_layer=1
padding=0

hidden_neurons_list_for_dense = [64]
activ_func_name_list_for_dense = [ActivFuncName.RELU.value, ActivFuncName.SOFTMAX.value]
weights_init_types_list_for_dense = [WeightsInitFuncName.HE.value]

training_loss_func_name = LossFuncName.CATEGORICAL_CROSS_ENTROPY.value

epochs = 20
learning_rate_conv = 0.001
learning_rate_dense = 0.001

is_save = True


"""
========================
CustomCnn WORKS BAD
========================
"""

# TRAIN
custom_cnn = CustomCnn(problem_name=problem_name, input_data=input_data, target=target, filters_num=filters_num, filter_size=filter_size, pool_size=pool_size,
                       hidden_neurons_list_for_dense=hidden_neurons_list_for_dense, activ_func_name_list_for_dense=activ_func_name_list_for_dense, weights_init_types_list_for_dense=weights_init_types_list_for_dense, training_loss_func_name=training_loss_func_name,
                       filter_init_func_name=filter_init_func_name, feature_map_activ_name=feature_map_activ_name,
                       stride_for_conv_layer=stride_for_conv_layer, stride_for_max_pooling_layer=stride_for_max_pooling_layer, padding=padding)

custom_cnn.train(epochs=epochs, learning_rate_conv=learning_rate_conv, learning_rate_dense=learning_rate_dense, is_save=True)


# custom_cnn = load_custom_cnn()
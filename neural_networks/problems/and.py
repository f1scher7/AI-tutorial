import numpy as np
from neural_networks.custom_nns_impl.custom_dense_multi_layers import CustomDenseMultiLayerNN, load_custom_dense_multilayer_nn
from env_loader import AND_SAVED_MODEL


# Data for AND problem (I KNOW THAT "AND PROBLEM" IS LINEAR :)
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [0], [0], [1]])

input_data_reshaped = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
target_reshaped = target.reshape(1, target.shape[0], target.shape[1])


problem_name = 'AND-problem'

data_norm_func_name = None
input_data_norm_params = None
target_norm_params = None

hidden_neurons_list = [4]
activ_funcs_list = ['sigmoid', 'sigmoid']
weights_init_types_list = ['xavier', 'xavier']

training_loss_func_name = 'mse'

epochs = 1000
learning_rate = 0.1
momentum = 0.9

reg_type = None
reg_lambda = 0.1


# INFERENCE
custom_dense_multilayer_nn = load_custom_dense_multilayer_nn(AND_SAVED_MODEL)

while True:
    a = int(input('A: '))
    b = int(input('B: '))

    user_input = np.array([[a, b]]).reshape(1, 1, 2)

    activated_output = custom_dense_multilayer_nn.inference(input_data=user_input)
    output_value = round(activated_output[0][0].item())

    print(f'AND({a}, {b}) = {output_value}')
    print("=====================================================================")


# TRAIN
# custom_dense_multilayer_nn = CustomDenseMultiLayerNN(
#     problem_name=problem_name, input_data_norm=input_data_reshaped, target_norm=target_reshaped,
#     data_norm_func_name=data_norm_func_name, input_data_norm_params=input_data_norm_params, target_norm_params=target_norm_params,
#     hidden_neurons_list=hidden_neurons_list, activ_funcs_list=activ_funcs_list, weights_init_types_list=weights_init_types_list,
#     training_loss_func_name=training_loss_func_name, epochs=epochs, learning_rate=learning_rate, momentum=momentum, reg_type=reg_type, reg_lambda=reg_lambda
# )
#
# custom_dense_multilayer_nn.train(is_save=True)
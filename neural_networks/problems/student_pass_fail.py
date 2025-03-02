import numpy as np
from neural_networks.custom_nn_types_impl.custom_dense_multi_layers import CustomDenseMultiLayerNN, load_custom_dense_multilayer_nn
from utils.math.data_normalization import min_max_normalization_for_each_feature, min_max_normalization_for_each_feature_with_params, min_max_denormalization_for_each_feature
from env_loader import STUDENT_PASS_FAIL_SAVED_MODEL


# First student: 3 hours of sleeping and 8 hours of studying
input_data = np.array([[3, 8], [4, 7], [8, 6], [7, 7], [6, 5], [2, 6], [1, 10],
                       [6, 4], [8, 0], [7, 2], [5, 1], [4, 5], [3, 5], [0, 5], [0, 10]])
# 0 - fail; 1 - pass
target = np.array([[1], [1], [1], [1], [1], [1], [0],
                   [1], [0], [0], [0], [1], [1], [0], [0]])

input_data_reshaped = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
target_reshaped = target.reshape(1, target.shape[0], target.shape[1])

input_data_norm, input_data_norm_params = min_max_normalization_for_each_feature(input_data_reshaped)


problem_name = 'STUDENT-PASS-FAIL-problem'

data_norm_func_name = 'min_max'
target_norm_params = None

hidden_neurons_list = [32]
activ_funcs_list = ['sigmoid', 'sigmoid']
weights_init_types_list = ['xavier', 'xavier']

training_loss_func_name = 'mse'

epochs = 3000
learning_rate = 0.1
momentum = 0.9

reg_type = None
reg_lambda = 0.1


# INFERENCE
custom_dense_multilayer_nn = load_custom_dense_multilayer_nn(STUDENT_PASS_FAIL_SAVED_MODEL)

while True:
    sleep_hours = int(input('Sleep hours: '))
    study_hours = int(input('Study hours: '))

    user_input = np.array([[sleep_hours, study_hours]]).reshape(1, 1, 2)
    norm_data = min_max_normalization_for_each_feature_with_params(user_input, custom_dense_multilayer_nn.input_data_norm_params)

    activated_output = custom_dense_multilayer_nn.inference(norm_data)
    output_value = round(activated_output[0][0].item())

    print(f'Predicted probability of passing for the student with {sleep_hours} hours of sleeping and {study_hours} hours of studying is: {activated_output[0][0].item():.2f}')

    if output_value >= 0.50:
        print('Student Pass!')
    else:
        print('Student Fail!')

    print('=====================================================================')


# TRAIN
# custom_dense_multilayer_nn = CustomDenseMultiLayerNN(
#     problem_name=problem_name, input_data_norm=input_data_norm, target_norm=target_reshaped,
#     data_norm_func_name=data_norm_func_name, input_data_norm_params=input_data_norm_params, target_norm_params=target_norm_params,
#     hidden_neurons_list=hidden_neurons_list, activ_funcs_list=activ_funcs_list, weights_init_types_list=weights_init_types_list,
#     training_loss_func_name=training_loss_func_name, epochs=epochs, learning_rate=learning_rate, momentum=momentum, reg_type=reg_type, reg_lambda=reg_lambda
# )
#
# custom_dense_multilayer_nn.train(is_save=True)


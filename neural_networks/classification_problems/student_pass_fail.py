import numpy as np

from utils.math_utils import min_max_normalization
from neural_networks.nn_types_impl import perceptron_nn_core
from neural_networks.nn_types_impl import single_layer_nn_core


norm_range = (0, 1)

# First student: 3 hours of sleeping and 5 hours of studying
X, min_val_norm, max_val_norm = min_max_normalization(
    np.array([[3, 8], [4, 7], [8, 6], [7, 7], [6, 5], [2, 6], [1, 10],
            [6, 4], [8, 0], [7, 2], [5, 1], [4, 5], [3, 5], [0, 5], [0, 10]]), norm_range)

# 0 - fail; 1 - pass
y = np.array([[1], [1], [1], [1], [1], [1], [0],
              [1], [0], [0], [0], [1], [1], [0], [0]])


epochs = 5000
learning_rate = 0.1
hidden_neurons = 4  # for single layer nn

# input_output_weights, bias_output_weights = perceptron_nn_core.train_perceptron_nn(X, y, epochs, learning_rate)
input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights = single_layer_nn_core.train_single_layer_nn(X, y, hidden_neurons, epochs, learning_rate)

while True:
    sleep_hours = int(input('Sleep hours: '))
    study_hours = int(input('Study hours: '))

    user_input, _, _ = min_max_normalization(np.array([[sleep_hours, study_hours]]), norm_range, min_val_norm, max_val_norm)

    # nn_output = perceptron_nn_core.predict_perceptron_nn(user_input, input_output_weights, bias_output_weights)
    nn_output = single_layer_nn_core.predict_single_layer_nn(user_input, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights)

    print(f'Predicted probability of passing for the student with {sleep_hours} hours of sleeping and {study_hours} hours of studying is: {nn_output[0][0]:.4f}')

    if nn_output[0][0] >= 0.50:
        print('Student Pass!')
    else:
        print('Student Fail!')

    print('=====================================================================')

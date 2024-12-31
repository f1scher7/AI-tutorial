import numpy as np

from utils.math.input_data_normalization import min_max_normalization
from neural_networks.nn_types_impl import perceptron
from neural_networks.nn_types_impl import single_layer
from utils.visualisation import create_window_plot


normalization_range = (0, 1)

# First student: 3 hours of sleeping and 5 hours of studying
X, min_val_norm, max_val_norm = min_max_normalization(
    np.array([[3, 8], [4, 7], [8, 6], [7, 7], [6, 5], [2, 6], [1, 10],
            [6, 4], [8, 0], [7, 2], [5, 1], [4, 5], [3, 5], [0, 5], [0, 10]]), normalization_range)
# 0 - fail; 1 - pass
y = np.array([[1], [1], [1], [1], [1], [1], [0],
              [1], [0], [0], [0], [1], [1], [0], [0]])


plot = create_window_plot('student_pass_fail training data', 10, 6)

pass_points = plot.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c="blue", s=50, label="Pass (1)", zorder=5)
fail_points = plot.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c="red", s=50, label="Fail (0)", zorder=5)

plot.xlabel('Hours of sleeping')
plot.ylabel('Hours of studying')
plot.title('Training data plot for student_pass_fail problem')
plot.legend(loc="upper right")
plot.grid(True, zorder=0)


epochs = 5000
learning_rate = 0.1
hidden_neurons = 8  # for single layer nn
input_to_hidden_weights_init_name = 'xavier'
hidden_to_output_weights_init_name = 'xavier'
hidden_activation_func_name = 'sigmoid'
output_activation_func_name = 'sigmoid'

# input_output_weights, bias_output_weights = perceptron_nn_core.train_perceptron_nn(X, y, epochs, learning_rate)
input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights = single_layer.train_single_layer_nn(X, y, epochs, learning_rate, hidden_neurons, input_to_hidden_weights_init_name, hidden_to_output_weights_init_name, hidden_activation_func_name, output_activation_func_name, plot)

while True:
    sleep_hours = int(input('Sleep hours: '))
    study_hours = int(input('Study hours: '))

    user_input, _, _ = min_max_normalization(np.array([[sleep_hours, study_hours]]), normalization_range, min_val_norm, max_val_norm)

    # nn_output = perceptron_nn_core.predict_perceptron_nn(user_input, input_output_weights, bias_output_weights)
    nn_output = single_layer.predict_single_layer_nn(user_input, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hidden_activation_func_name, output_activation_func_name)

    print(f'Predicted probability of passing for the student with {sleep_hours} hours of sleeping and {study_hours} hours of studying is: {nn_output[0][0]:.4f}')

    if nn_output[0][0] >= 0.50:
        print('Student Pass!')
    else:
        print('Student Fail!')

    print('=====================================================================')

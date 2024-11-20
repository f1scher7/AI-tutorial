import numpy as np

from neural_networks.nn_types_impl import perceptron


# Data for AND problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

epochs = 5000
learning_rate = 0.1
hidden_to_output_weights_init_name = 'random'
output_activation_func_name = 'sigmoid'

input_to_output_weights, bias_output_weights = perceptron.train_perceptron_nn(X, y, epochs, learning_rate, hidden_to_output_weights_init_name, output_activation_func_name)

print('AND problem')

while True:
    a = int(input('A: '))
    b = int(input('B: '))

    user_input = np.array([[a, b]])

    nn_output = perceptron.predict_perceptron_nn(user_input, input_to_output_weights, bias_output_weights, output_activation_func_name)

    print(f'AND({a}, {b}) = {nn_output[0][0]:.4f}')
    print("=====================================================================")
import numpy as np

from neural_networks.nn_types_impl import single_layer
from utils.visualisation import create_window_plot


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


plot = create_window_plot('XOR training data', 10, 6)

pass_points = plot.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c="blue", s=50, label="XOR = 1", zorder=5)
fail_points = plot.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c="red", s=50, label="XOR = 0", zorder=5)

plot.xlabel('A')
plot.ylabel('B')
plot.title('Training data plot for XOR problem')
plot.legend(loc="upper right")
plot.grid(True, zorder=0)


epochs = 5000
learning_rate = 0.01
hidden_neurons = 4 # for single layer nn
input_to_hidden_weights_init_name = 'random'
hidden_to_output_weights_init_name = 'random'
hidden_activation_func_name = 'relu'
output_activation_func_name = 'relu'

input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights = single_layer.train_single_layer_nn(X, y, epochs, learning_rate, hidden_neurons, input_to_hidden_weights_init_name, hidden_to_output_weights_init_name, hidden_activation_func_name, output_activation_func_name, plot)

print('XOR problem')

while True:
    a = int(input('A: '))
    b = int(input('B: '))

    user_input = np.array([[a, b]])

    nn_output = single_layer.predict_single_layer_nn(user_input, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hidden_activation_func_name, output_activation_func_name)

    print(f'XOR({a}, {b}) = {nn_output[0][0]:.4f}')
    print("=====================================================================")

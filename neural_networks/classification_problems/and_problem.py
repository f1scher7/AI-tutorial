import numpy as np
import perceptron_nn_core


# data for AND problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

epochs = 5000
learning_rate = 0.1

input_to_output_weights, bias_output_weights = perceptron_nn_core.train_perceptron_nn(X, y, epochs, learning_rate)

print('AND problem')

while True:
    a = int(input('A: '))
    b = int(input('B: '))

    user_input = np.array([[a, b]])

    nn_output = perceptron_nn_core.predict(user_input, input_to_output_weights, bias_output_weights)

    print(f'AND({a}, {b}) = {nn_output[0][0]:.4f}')
    print("=====================================================================")
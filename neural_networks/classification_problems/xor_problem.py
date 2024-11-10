import numpy as np
import time

from utils import math_utils
from utils import displaying_nn_utils


def xor_nn_predict(input_to_hidden_weights_f, hidden_to_output_weights_f, )

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

input_to_hidden_weights = np.random.uniform(low=-1, high=1, size=(2, 6))
hidden_to_output_weights = np.random.uniform(low=-1, high=1, size=(6, 1))

bias_hidden_weights = np.random.uniform(low=-1, high=1, size=(1, 6))
bias_output_weights = np.random.uniform(low=-1, high=1, size=(1, 1))

final_nn_output = np.array([])
mse_values = []

epochs = 5000
learning_rate = 0.1

start_time = time.perf_counter()

for epoch in range(epochs):
    # Forward propagation
    pre_activated_hidden = np.dot(X, input_to_hidden_weights) + bias_hidden_weights # Potted sum for hidden layer
    activated_hidden = math_utils.sigmoid(pre_activated_hidden) # Neurons activation for hidden layer

    pre_activation_output = np.dot(activated_hidden, hidden_to_output_weights) + bias_output_weights
    activated_output = math_utils.sigmoid(pre_activation_output)

    final_nn_output = activated_output
    mse_values.append(math_utils.mean_squared_error(y, activated_output))

    # Back propagation
    output_error = y - activated_output
    # Output gradient - CORRECTIONS for each component
    # Output error * sensitivity for each component = how each component contributed to output error
    output_gradient = output_error * math_utils.sigmoid_derivative(activated_output)

    hidden_error = np.dot(output_gradient, hidden_to_output_weights.T)
    hidden_gradient = hidden_error * math_utils.sigmoid_derivative(activated_hidden)

    # Update weights and biases
    hidden_to_output_weights += np.dot(activated_hidden.T, output_gradient) * learning_rate
    input_to_hidden_weights += np.dot(X.T, hidden_gradient) * learning_rate

    bias_output_weights += np.sum(output_gradient, axis=0, keepdims=True)
    bias_hidden_weights += np.sum(hidden_gradient, axis=0, keepdims=True)

end_time = time.perf_counter()
training_time = end_time - start_time

displaying_nn_utils.print_result_nn(final_nn_output, epochs, training_time)
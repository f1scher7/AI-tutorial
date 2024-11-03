import numpy as np
import matplotlib.pyplot as plt
import time
import utils

np.random.seed(42)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

input_to_hidden_weights = np.random.uniform(low=-1, high=1, size=(2, 3))
hidden_to_output_weights = np.random.uniform(low=-1, high=1, size=(3, 1))

bias_hidden_weights = np.random.uniform(low=-1, high=1, size=(1, 3))
bias_output_weights = np.random.uniform(low=-1, high=1, size=(1, 1))

learning_rate = 0.1
epochs = 10000

mse_values = []

start_time = time.perf_counter()

for epoch in range(epochs):
    hidden_input = np.dot(X, input_to_hidden_weights) + bias_hidden_weights
    hidden_output = utils.sigmoid(hidden_input)

    # print(f'x:\n{x}')
    # print(f'input_to_hidden_weights:\n{input_to_hidden_weights}')
    # print(f'dot(x, input_to_hidden_weights):\n{np.dot(x, input_to_hidden_weights)}')
    # print(f'bias_hidden_weights:\n{bias_hidden_weights}')
    # print(f'dot(x, input_to_hidden_weights) + bias_hidden_weights:\n{np.dot(x, input_to_hidden_weights) + bias_hidden_weights}')
    # print(f'hidden_output:\n{hidden_output}')

    final_input = np.dot(hidden_output, hidden_to_output_weights) + bias_output_weights
    final_output = utils.sigmoid(final_input)

    print(f'final_output:\n{final_output}')

    error = y - final_output
    mse_values.append(utils.mean_squared_error(y, final_output))

    d_final_output = error * utils.sigmoid_derivative(final_output)
    print(f'd_final_output:\n{d_final_output}')

    error_hidden_layer = d_final_output.dot(hidden_to_output_weights.T)
    d_hidden_output = error_hidden_layer * utils.sigmoid_derivative(hidden_output)

    hidden_to_output_weights += hidden_output.T.dot(d_final_output) * learning_rate
    input_to_hidden_weights += X.T.dot(d_hidden_output) * learning_rate

    bias_output_weights += np.sum(d_final_output, axis=0, keepdims=True) * learning_rate
    bias_hidden_weights += np.sum(d_hidden_output, axis=0, keepdims=True) * learning_rate

    # if epoch % 1000 == 0:
    #     print(f'Epoch {epoch}\nWeights:\n{input_to_hidden_weights}\nFinal Output:\n{final_output}\nMSE: {mse}')
    #     print("==============================================================")

end_time = time.perf_counter()
training_time = end_time - start_time

# print(f'Training time: {training_time:.2f} secs')
# print(f'Results after {epochs} epochs:\n{final_output}')
#
plt.plot(mse_values)
plt.title("MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.grid()
plt.show()
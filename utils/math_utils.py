import numpy as np

def sigmoid(x_f):
    return 1 / (1 + np.exp(-x_f))

def sigmoid_derivative(x_f):
    return x_f * (1 - x_f)

def mean_squared_error(y_true_f, y_pred_f):
    return np.mean((y_true_f - y_pred_f) ** 2)

def print_result_nn(training_time_f, final_output_f, epochs_f):
    print(f'Training time: {training_time_f:.2f} secs')
    print(f'Result after {epochs_f} epochs:\n{final_output_f}')
    print('=====================================================================')

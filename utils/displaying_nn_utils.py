import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from neural_networks.nn_types_impl.single_layer_nn_core import predict_single_layer_nn

def create_window_plot(window_title, size_x, size_y):
    matplotlib.use('TkAgg')

    plt.figure(figsize=(size_x, size_y))

    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title(window_title)

    return plt


def plot_decision_boundary(X_f, input_to_hidden_weights_f, hidden_to_output_weights_f, bias_hidden_weights_f, bias_output_weights_f, epoch_f, plot_f):
    x_min, x_max = X_f[:, 0].min() - 1, X_f[:, 0].max() + 1
    y_min, y_max = X_f[:, 1].min() - 1, X_f[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_preds = predict_single_layer_nn(grid_points, input_to_hidden_weights_f, hidden_to_output_weights_f,bias_hidden_weights_f, bias_output_weights_f)
    zz = grid_preds.reshape(xx.shape)

    if hasattr(plot_f, 'contour_line'):
        plot_f.contour_line.collections[0].remove()

    plot_f.contour_line = plot_f.contour(xx, yy, zz, levels=[0.5], colors="green", linewidths=1)

    plot_f.suptitle(f"Epoch {epoch_f + 1}")
    plot_f.draw()


def plot_mse(mse_values_f):
    plt.plot(mse_values_f, label='MSE', color='blue')
    plt.title('Mean Squared Error over Epochs')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()


def print_result_nn(final_output_f, epochs_f, training_time_f):
    print(f'Training time: {training_time_f:.2f} secs')
    print(f'Result after {epochs_f} epochs:\n{final_output_f}')
    print("=====================================================================")

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def plot_training_losses(training_loss_function_name, training_losses):
    matplotlib.use('TkAgg')

    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title(training_loss_function_name.upper())

    plt.plot(training_losses, label='Losses', color='blue')
    plt.title(f'{training_loss_function_name.upper()} Loss over Epochs')
    plt.ylabel(f'{training_loss_function_name.upper()}')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()


def print_training_logs_nn(nn_name, epoch, epochs, training_loss, training_loss_function_name):
    print(nn_name)
    print(f'Epoch: {epoch}/{epochs}')
    print(f'Training loss ({training_loss_function_name}): {training_loss}')
    print("=====================================================================")


# def create_window_plot(window_title, size_x, size_y):
#     matplotlib.use('TkAgg')
#
#     plt.figure(figsize=(size_x, size_y))
#
#     fig_manager = plt.get_current_fig_manager()
#     fig_manager.set_window_title(window_title)
#
#     return plt
#
#
# # def plot_decision_boundary(x, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hid_act_func_name, out_act_func_name, epoch, plot):
# #     x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
# #     y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
# #
# #     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
# #     grid_points = np.c_[xx.ravel(), yy.ravel()]
# #
# #     grid_preds = predict_single_layer_nn(grid_points, input_to_hidden_weights, hidden_to_output_weights, bias_hidden_weights, bias_output_weights, hid_act_func_name, out_act_func_name)
# #     zz = grid_preds.reshape(xx.shape)
# #
# #     if hasattr(plot, 'contour_line'):
# #         plot.contour_line.collections[0].remove()
# #
# #     plot.contour_line = plot.contour(xx, yy, zz, levels=[0.5], colors="green", linewidths=1)
# #
# #     plot.suptitle(f"Epoch {epoch + 1}")
# #     plot.draw()
# #
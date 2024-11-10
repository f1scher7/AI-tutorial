from matplotlib import pyplot as plt


def print_result_nn(final_output_f, epochs_f, training_time_f):
    print(f'Training time: {training_time_f:.2f} secs')
    print(f'Result after {epochs_f} epochs:\n{final_output_f}')
    print("=====================================================================")


def plot_mse(mse_values_f):
    plt.plot(mse_values_f, label='MSE', color='blue')
    plt.title('Mean Squared Error over Epochs')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.show()
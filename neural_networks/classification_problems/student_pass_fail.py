import numpy as np
import perceptron_nn_core

#First student: 3 hours of sleeping and 5 hours of studying
X = np.array([[5, 1], [1, 5], [8, 0], [10, 0], [5, 2], [3, 3], [4, 4], [7, 3], [5, 3], [6, 2], [4, 5]])
#0 - fail; 1 - pass
y = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

epochs = 5000
learning_rate = 0.1

input_output_weights, bias_output_weights = perceptron_nn_core.train_perceptron_nn(X, y, epochs, learning_rate)

while True:
    sleep_hours = int(input('Sleep hours: '))
    study_hours = int(input('Study hours: '))

    user_input = np.array([[sleep_hours, study_hours]])

    nn_output = perceptron_nn_core.predict(user_input, input_output_weights, bias_output_weights)

    print(f'Predicted probability of passing for the student with {sleep_hours} hours of sleeping and {study_hours} hours of studying is: {nn_output[0][0]:.4f}')

    if nn_output[0][0] >= 0.50:
        print('Student Pass!')
    else:
        print('Student Fail!')

    print('=====================================================================')

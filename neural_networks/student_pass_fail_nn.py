import numpy as np
import matplotlib.pyplot as plt
import time
import utils

def is_student_pass_fail(sleep_hours, study_hours):
    student_data = np.array([sleep_hours, study_hours])
    student_final_input = np.dot(student_data, input_to_output_weights) + bias_output_weights
    student_final_output = utils.sigmoid(student_final_input)

    print(f'Predicted probability of passing for the student with {sleep_hours} hours of sleeping and {study_hours} hours of studying is: {student_final_output[0][0]:.4f}')

    if student_final_output[0][0] >= 0.50:
        print('Student Pass!')
    else:
        print('Student Fail!')

    print('=====================================================================')

#First student: 3 hours of sleeping and 5 hours of studying
X = np.array([[5, 1],
              [8, 0],
              [5, 2],
              [3, 3],
              [4, 4],
              [7, 3],
              [5, 3],
              [6, 2],
              [4, 5]])

#0 - fail; 1 - pass
y = np.array([[0],
              [0],
              [0],
              [0],
              [1],
              [1],
              [1],
              [1],
              [1]])

input_to_output_weights = np.random.uniform(low=-1, high=1, size=(2, 1))
bias_output_weights = np.random.uniform(low=-1, high=-1, size=(1, 1))

np.random.seed(42)

epochs = 5000
learning_rate = 0.1

mse_values = []

start_time = time.perf_counter()

for epoch in range(epochs):
    #Forward propagation
    final_input = np.dot(X, input_to_output_weights) + bias_output_weights
    final_output = utils.sigmoid(final_input)

    error = y - final_output
    mse_values.append(utils.mean_squared_error(y, final_output))

    #Back propagation
    d_final_output = error * utils.sigmoid_derivative(final_output) #Gradient matrix

    input_to_output_weights += np.dot(X.T, d_final_output) * learning_rate
    bias_output_weights += np.sum(d_final_output, axis=0, keepdims=True) * learning_rate

end_time = time.perf_counter()
training_time = end_time - start_time

print(f'Training time: {training_time:.2f} secs')
print(f'Result after {epochs} epochs:\n{final_output}')
print('=====================================================================')

# plt.figure(figsize=(10, 5))
# plt.plot(mse_values, label='MSE', color='blue')
# plt.title('Mean Squared Error over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.legend()
# plt.grid()
# plt.show()

while True:
    student_sleep_hours = int(input('Sleep hours: '))
    student_study_hours = int(input('Study hours: '))

    is_student_pass_fail(student_sleep_hours, student_study_hours)
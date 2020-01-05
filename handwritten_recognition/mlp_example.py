import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

training_input = np.array([[0,0,1],
							[1,1,1],
							[1,0,1],
							[0,1,1]])

training_output = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1


print("Random starting synaptic weights: ")
print(synaptic_weights)

# N = 1000000
N = 100000


input_layer = training_input

for iteration in range(N):
	hidden_output = np.dot(input_layer, synaptic_weights)
	reverse_hidden_output = -1 * hidden_output
	output = sigmoid(hidden_output)
	# for each weight in synaptic_weights
	# calculate the derivative of the weight to the error
	# and update it by adding the original one
	# x = hidden layer output = np.dot(input_layer, synaptic_weights)
	# y_hat = output layer = output
	# x1 = input
	# 1) y_hat - y
	# 2) e^-x / (1+e^-x)^2
	# 3) x1
	output_delta = output - training_output
	activation_derivatives = np.exp(reverse_hidden_output) / pow(1 + np.exp(reverse_hidden_output), 2)
	# print(activation_derivatives)
	# print(output_delta)
	# print(np.dot(output_delta, activation_derivatives))
	delta_w = np.dot(training_input.T, activation_derivatives * output_delta)
	# print(delta_w)
	synaptic_weights -= delta_w

print('Output after training:')
print(output)

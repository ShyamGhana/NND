# 1. implementation of simple free forward neural network using numpy

import numpy as np

inputs = np.array([1, 2, 3])

weights1 = np.array([
    [4, 5],
    [6, 7],
    [8, 9]
])

bias1 = np.array([1, 2])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output = np.dot(inputs, weights1) + bias1
print("Output:", output)

activated_output = sigmoid(output)
print("Activated output:", activated_output)

activated_output2 = sigmoid(activated_output)
print("Activated output:", activated_output2)

weights2 = np.array([
    [0.1],
    [0.2]
])

bias2 = np.array([1])

final_output = np.dot(activated_output2, weights2) + bias2
final_output = sigmoid(final_output)
print("Final output:", final_output)

y = 1
loss = (y - final_output) ** 2
print("Loss:", loss)

lr = 0.1

dloss_da2 = -2 * (y - final_output)
da2_dz2 = final_output * (1 - final_output)
dz2_dw2 = activated_output2.reshape(2, 1)

grad_w2 = dloss_da2 * da2_dz2 * dz2_dw2
weights2 = weights2 - lr * grad_w2

print("Updated weights2:", weights2)


# OUTPUT 

Output: [41 48]
Activated output: [1. 1.]
Activated output: [0.73105858 0.73105858]
Final output: [0.77194343]
Loss: [0.0520098]
Updated weights2: [[0.1058702]
 [0.2058702]]

#implement free forward NN using numpy with class
Date : 9/01/2026
Day : Friday 

import numpy as np

class FeedForwardNeuralNetwork:
    def __init__(self):
        self.weights1 = np.array([[4, 5],
                                  [6, 7],
                                  [8, 9]], dtype=float)

        self.bias1 = np.array([1, 2], dtype=float)

        self.weights2 = np.array([[0.1],
                                  [0.2]], dtype=float)

        self.bias2 = np.array([1], dtype=float)

        self.lr = 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def compute_loss(self, y, y_pred):
        return (y - y_pred) ** 2

    def backward(self, x, y):
        dloss_da2 = -2 * (y - self.a2)
        da2_dz2 = self.sigmoid_derivative(self.a2)

        grad_w2 = dloss_da2 * da2_dz2 * self.a1.reshape(2, 1)
        grad_b2 = float(dloss_da2 * da2_dz2)

        dloss_da1 = dloss_da2 * da2_dz2 * self.weights2.T
        da1_dz1 = self.sigmoid_derivative(self.a1)

        grad_w1 = x.reshape(3, 1) @ (dloss_da1 * da1_dz1)
        grad_b1 = (dloss_da1 * da1_dz1).reshape(2,)

        self.weights2 -= self.lr * grad_w2
        self.bias2 -= self.lr * grad_b2

        self.weights1 -= self.lr * grad_w1
        self.bias1 -= self.lr * grad_b1

    def train(self, x, y, epochs=1000):
        for i in range(epochs):
            y_pred = self.forward(x)
            loss = self.compute_loss(y, y_pred)
            self.backward(x, y)

            if (i + 1) % 100 == 0:
                print("Epoch:", i + 1, "Loss:", loss)

        return y_pred, loss


x = np.array([1, 2, 3], dtype=float)
y = 1

model = FeedForwardNeuralNetwork()

output = np.dot(x, model.weights1) + model.bias1
print("Output:", output)

activated_output = model.sigmoid(output)
print("Activated output:", activated_output)

activated_output2 = model.sigmoid(activated_output)
print("Activated output:", activated_output2)

final_output, final_loss = model.train(x, y, epochs=1000)

print("Final output:", final_output)
print("Loss:", final_loss)
print("Updated weights2:", model.weights2)


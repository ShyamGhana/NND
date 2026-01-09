# Day 1

Day : Monday 
Date : 05/01/2026

Project 1 : Spam Prediction Neural Network

import math

# Activation Functions
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Inputs 
x1, x2, x3 = 1, 0, 1


# Weights (Input → Hidden)
w1_h1, w2_h1, w3_h1 = 0.5, 0.2, 0.3
w1_h2, w2_h2, w3_h2 = 0.4, 0.1, -0.5


# Hidden Layer Calculation 
h1_input = (x1 * w1_h1) + (x2 * w2_h1) + (x3 * w3_h1)
h2_input = (x1 * w1_h2) + (x2 * w2_h2) + (x3 * w3_h2)

h1_output = relu(h1_input)
h2_output = relu(h2_input)

print("H1 after ReLU:", h1_output)
print("H2 after ReLU:", h2_output)


# Weights (Hidden → Output) 
w_h1_out = 0.7
w_h2_out = 0.2


# Output Layer 
output_input = (h1_output * w_h1_out) + (h2_output * w_h2_out)
final_output = sigmoid(output_input)

print("Output before Sigmoid:", output_input)
print("Final Spam Probability:", final_output)

#OUTPUT 

H1 after ReLU: 0.8
H2 after ReLU: 0
Output before Sigmoid: 0.5599999999999999
Final Spam Probability: 0.6364525402815664


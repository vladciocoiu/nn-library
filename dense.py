from layer import Layer
import numpy as np

class Dense(Layer):
    # initialize weights and biases with random uniformly distributed numbers
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.biases = np.random.rand(output_size, 1) - 0.5

    # pass the input through the layer, returning the output
    def forward_prop(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    # computes the gradient for each weight and bias (using the chain rule) and updates them accordingly
    def backward_prop(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= weight_gradient * learning_rate

        self.biases -= output_gradient * learning_rate

        return np.dot(self.weights.T, output_gradient)

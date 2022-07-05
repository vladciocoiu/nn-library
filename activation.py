from layer import Layer
import numpy as np

class Activation(Layer):

    # initialize activation layer with the activation function and its derivative (used for back propagation)
    def __init__(self, function, function_prime):
        self.function = function
        self.function_prime = function_prime

    # just pass the input through the function
    def forward_prop(self, input):
        self.input = input
        return self.function(self.input)

    # get the gradient of the input by multiplying the output gradient with the derivative of the function (chain rule)
    def backward_prop(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.function_prime(self.input))

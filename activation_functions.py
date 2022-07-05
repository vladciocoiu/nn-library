from activation import Activation
import numpy as np

# Sigmoid function s(x) = 1 / (1 + e^(-x))
# with its derivative s'(x) = s(x) * (1 - s(x))
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)

# Hyperbolic Tangent function 
# tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)), but we don't need to compute that since numpy has a builtin tanh function
# its derivative tanh'(x) = 1 - tanh^2(x)
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, tanh_prime)
        
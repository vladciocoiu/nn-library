import numpy as np

# base layer class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_prop(self, input):
        pass

    def backward_prop(self, output_gradient, learning_rate):
        pass
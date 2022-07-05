import numpy as np

# wrapper class for the layers and the train and predict functions
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    # just pass the input sequentially through all the layers
    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output

    # function that trains our network using backpropagation
    def train(self, loss, loss_prime, input, output, epochs, learning_rate):
        for i in range(epochs):
            error = 0

            # go through all input examples and improve the network after each one
            for x, y in zip(input, output):
                ans = self.predict(x)
                
                # calculate the error using the loss function (for display only)
                error += loss(ans, y)
                
                # gradient of the output using the derivative of the loss function
                gradient = loss_prime(ans, y)
                
                # propagate backwards, adjusting each weight and bias
                for layer in reversed(self.layers):
                    gradient = layer.backward_prop(gradient, learning_rate)
                    
            error /= len(input)
                
            print(f'{i + 1}/{epochs}, error={error}')
import numpy as np

from neural_network import NeuralNetwork
from dense import Dense
from activation_functions import Sigmoid
from loss_functions import mse, mse_prime

# ----- training a model to solve XNOR -----

# create a network with 1 hidden layer, consisting of 3 nodes
network = NeuralNetwork([
    Dense(2, 3),
    Sigmoid(),
    Dense(3, 1),
    Sigmoid()
])

# create input and output data
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1)
output = np.array([1, 0, 0, 1]).reshape(4, 1, 1)

# train the model
network.train(mse, mse_prime, input, output, 10000, 0.1)

# get the prediction for an operation
op = [[1], [1]]
print(f'Operands: {op[0][0]}, {op[1][0]}')
[[ans]] = network.predict(np.array(op))
print(f'Ans={int(ans >= 0.5)}')

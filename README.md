# nn-library

This is a simple Neural Network library made for learning purposes

## Functions

#### Layer types
``Dense(input_size, output_size)`` creates a new dense layer with specific input and output sizes, initialized with random weight and bias values
* A _dense layer_ computes a _linear combination_ of its inputs and weights and biases

``Activation(function, function_prime)`` creates a new activation layer with a specific activation function and its derivative
* An _activation layer_ passes the input through the _activation function_
* This library has 2 builtin activation layers: ``Tanh()`` and ``Sigmoid()``

#### Creating a NeuralNetwork object
``NeuralNetwork(layers)`` creates a new NeuralNetwork object using a list of layers

#### Training and predicting
``NeuralNetwork.train(loss, loss_prime, input, output, epochs, learning_rate)`` trains the model using gradient descent and backpropagation
* ``input`` should be a numpy array with a ``(x, NeuralNetwork.input_size, 1)`` shape, where x is the number of training examples
* ``output`` should have a shape of ``(x, NeuralNetwork.output_size, 1)``

``NeuralNetwork.predict(input)`` returns the prediction of the model based on the input

#### Loss Functions
Currently, there is only one builtin loss function, the mean squared error ``mse(ans, y)``

## Usage

#### Cloning the repo
```
git clone https://github.com/vladciocoiu/nn-library.git
cd nn-library
```

#### Importing the library
```
from neural_network import NeuralNetwork
from dense import Dense
from activation import Activation
from activation_functions import Sigmoid, Tanh
from loss_functions import mse, mse_prime
```

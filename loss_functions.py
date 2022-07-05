import numpy as np

# mean squared error is the average of the squared differences between the real answer and our answer
def mse(ans, y):
    return np.mean(np.power(ans - y, 2))

# derivative of mse with respect to y
def mse_prime(ans, y):
    return 2 * (ans - y) / np.size(ans)

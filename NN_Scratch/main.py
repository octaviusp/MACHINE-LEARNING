from generator.perceptron import test_create_perceptron
from generator.activation_functions import Relu
from dtypes.floats import FLOAT_16
from generator.layers import test_forward_propagation

import numpy as np

np.random.seed(1)

size = 5
perceptrons = []
for i in range(size):
    perceptrons.append(test_create_perceptron(Relu(), FLOAT_16(1), FLOAT_16, 3, np.random.rand(3)))

print(repr(perceptrons))

test_forward_propagation(perceptrons)


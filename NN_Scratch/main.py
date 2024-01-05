from generator.perceptron import test_create_perceptron
from generator.activation_function import Relu, Sigmoid, TanH
from dtypes.floats import FLOAT_16
from generator.layer import test_forward_propagation, test_deep_network

import numpy as np

np.random.seed(1)

size = 2
perceptrons = []
for i in range(size):
    perceptrons.append(test_create_perceptron(Relu() if i == 0 else Sigmoid(), FLOAT_16(0.1), FLOAT_16, 3, np.random.rand(3)))

print(repr(perceptrons))

test_forward_propagation(perceptrons)

print("- TESTING DEEP NETWORK ")

print(perceptrons)

perceptrons_layer_0 = []
for i in range(3):
    perceptrons_layer_0.append(test_create_perceptron(Relu() if i == 0 else Sigmoid(), FLOAT_16(0.1), FLOAT_16, 3, np.random.rand(3)))

perceptrons_layer_1 = []
for i in range(2):
    perceptrons_layer_1.append(test_create_perceptron(Relu() if i == 0 else Sigmoid(), FLOAT_16(0.1), FLOAT_16, 3, np.random.rand(3)))

perceptrons_layer_2 = []
for i in range(1):
    perceptrons_layer_2.append(test_create_perceptron(Relu() if i == 0 else Sigmoid(), FLOAT_16(0.1), FLOAT_16, 2, np.random.rand(2)))

perceptrons = [perceptrons_layer_0, perceptrons_layer_1, perceptrons_layer_2]

test_deep_network(perceptrons)


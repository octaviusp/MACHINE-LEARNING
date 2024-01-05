from dtypes.dtype import DTYPE
from dtypes.floats import FLOAT_16, FLOAT_32
from generator.activation_function import ActivationFunction, Relu, Sigmoid, TanH

import numpy as np

class Perceptron():
    """
        Rosenblatt Perceptron base class.
        Forward propagation mean computing z therefore activation functions.
    """
    def __init__(self, dtype: DTYPE, bias: DTYPE, activation_function: ActivationFunction, input_size: int, initial_weights: np.array):
        self.input_size = input_size
        self.W = initial_weights
        self.activation_function = activation_function
        self.dtype = dtype
        self.bias = bias
    
    def restart_W(self):
        self.W = np.zeros(self.input_size)

    def update_W_size(self, new_W_size: int):
        self.W = np.zeros(new_W_size)

    def update_W(self, update_W):
        self.W = update_W

    def compute_z(self, input):
        return self.dtype(np.dot(self.W, input)) + self.bias
    
    def compute_a(self, z: DTYPE):
        return self.activation_function.activate(z)

    def __repr__(self) -> str:
        return f"""
            "Perceptron":
                "weights": {self.W},
                "bias": {self.bias},
                "activation_function": {self.activation_function.name},
                "dtype": {self.dtype}
            """

def test_create_perceptron(activation_function: ActivationFunction, bias: DTYPE, type: DTYPE, input_size: int, initial_weights: np.array):
    return Perceptron(type, bias, activation_function, input_size, initial_weights)

def test_linear_regression():
    relu = Relu()
    p1 = Perceptron(FLOAT_16, FLOAT_16(450), relu, 1, initial_weights=np.array([10, 200, 12]))
    houses = np.array([[100, 2, 13], [87, 3, 11], [50, 4, 14]])
    for house in houses:
        p1_z = p1.compute_z(house)
        p1_a = p1.compute_a(p1_z)
        print("Perceptron Linear regresion output: ", p1_a)

def basic_test():
    relu = Relu()
    sigmoid = Sigmoid()
    tanh = TanH()
    p1 = Perceptron(FLOAT_16, FLOAT_16(1), tanh, 1, initial_weights=np.array([-1,-2]))
    print("Perceptron created. AF: ", p1.activation_function)
    p1_z = p1.compute_z(np.array([1,2]))
    print("Perceptron P1 computed z: ", p1_z)
    p1_a = p1.compute_a(p1_z)
    print("Perceptron p1 computed a: ", p1_a)
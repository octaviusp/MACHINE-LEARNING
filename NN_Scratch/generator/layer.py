from .perceptron import Perceptron

import numpy as np

class Layer():

    def __init__(self, post_layer, prev_layer_output: np.array, hidden_units: list[Perceptron]):
        self.hidden_units = hidden_units
        self.post_layer = post_layer
        self.prev_layer_output = prev_layer_output
    
    def forward_propagation(self):
        for n, hidden_unit in enumerate(self.hidden_units):
            z = hidden_unit.compute_z(self.prev_layer_output)
            a = hidden_unit.compute_a(z)
            print(f"Computed a{n}: ", a)

def test_forward_propagation(perceptrons):
    inputs = np.array([12, 13, 14])
    layer_1 = Layer(None, inputs, perceptrons)
    layer_1.forward_propagation()
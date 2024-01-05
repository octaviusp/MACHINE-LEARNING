from .perceptron import Perceptron

import numpy as np

class Layer():

    def __init__(self, hidden_units: list[Perceptron]):
        self.hidden_units = hidden_units
    
    def forward_propagation(self, input):
        outputs = []
        print(self.hidden_units)
        for n, hidden_unit in enumerate(self.hidden_units):
            z = hidden_unit.compute_z(input)
            a = hidden_unit.compute_a(z)
            outputs.append(a)
            print(f"Computed a{n}: ", a)
        return outputs

def test_forward_propagation(perceptrons):
    inputs = np.array([12, 13, 14])
    layer_1 = Layer(None, inputs, perceptrons)
    layer_1.forward_propagation()

def test_deep_network(perceptrons):
    # only one training example for sake of simplicity
    simulate_inputs = np.array([12, 13, 14])
    layers_size = 3
    last_output = simulate_inputs
    for i in range(layers_size):
        print("Layer ", i, " computations")
        # perceptrons[i] will give me the respective perceptrons for the current layer
        layer = Layer(None, last_output, perceptrons[i])
        outputs = layer.forward_propagation()
        last_output = outputs
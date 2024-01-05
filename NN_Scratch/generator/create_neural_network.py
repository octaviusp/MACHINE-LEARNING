from dtypes.dtype import DTYPE
from .layer import Layer
from .perceptron import Perceptron

import numpy as np

class NN_Generator():

    def __init__(self, dtype: DTYPE, hidden_layers: int):
        self.dtype = dtype
        self.hidden_layers = hidden_layers
        self.layers_order: list[Layer] = []
        self.first_layer_input_size = None
        self.last_output_size = None
    
    def initialize_layers(self, input_size_first_layer: np.array, perceptrons: list[list[Perceptron]]):
        if not len(perceptrons) == self.hidden_layers:
            raise Exception("Perceptrons list must be equal to the size of layers in the network")
        for i in range(self.hidden_layers):
            if i == 0:
                self.first_layer_input_size = input_size_first_layer
            layer = Layer(perceptrons[i])
            self.last_output_size = np.array(np.zeros(len(layer.hidden_units)))
            self.layers_order.append(layer)
    
    def see_architecture(self):
        return self.hidden_layers
    
    def predict(self, inputs: np.array):
        """ if not  inputs.shape == self.input_size_first_layer.shape:
                    raise Exception("The training example can't be predicted because the features input isn't what network is awaiting.")
                """
        last_output = None
        for i, layer in enumerate(self.layers_order):
            last_output = layer.forward_propagation(inputs) if i == 0 else layer.forward_propagation(last_output)
            print(f"Layer {i} output: ", last_output)
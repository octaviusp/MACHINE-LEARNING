from dtypes.dtype import DTYPE
import numpy as np

class ActivationFunction():

    def __init__(self, name: str):
        self.name = name
    
    def activate(z: DTYPE):
        return z
    
    def __repr__(self) -> str:
        return self.name

class Relu(ActivationFunction):
    def __init__(self):
        super().__init__(name="relu")
    
    def activate(self, z: DTYPE):
        return 0 if z <= 0 else z
    
    def derivative(self, z: DTYPE):
        return 0 if z <= 0 else 1

class Sigmoid(ActivationFunction):

    def __init__(self):
        super().__init__(name="sigmoid")
    
    def activate(self, z: DTYPE):
        return 1 / ( 1 + np.exp(-z) )
    
    def derivative(self, z: DTYPE):
        evaluation = self.activate(z)
        return evaluation*(1-evaluation)

class TanH(ActivationFunction):
    def __init__(self):
        super().__init__(name="tanh")
    
    def activate(self, z: DTYPE):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    def derivative(self, z:DTYPE):
        return 1 - self.activate(z)**2
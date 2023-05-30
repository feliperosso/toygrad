"""
layers.py

Construct some of the basic layers using the Tensor class
defined in engine.py of the toygrad package
"""

# Load packages
import math
import numpy as np
from toygrad.engine import Tensor

# - Define the layers parent class -
class nnModule:

    def __init__(self):
        # Store learnable parameteres
        self.parameters = []
        # Dropout train vs eval model marker
        self.training = True
    
    def __call__(self, *input):
        return self.forward(*input)
        
    def addParameter(self, parameter):
        """ Store the input 'parameter'."""
        self.parameters += [parameter]
        return parameter

# - Dropout -
class Dropout(nnModule):

    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate # If dropout_rate = 0, mask = 1

    def forward(self, input:Tensor) -> Tensor:
        if self.training:
            mask = np.random.uniform(0, 1, input.item.shape) > self.dropout_rate
            mask = Tensor(mask/(1 - self.dropout_rate))
            return input*mask
        else:
            return input

# - Linear -
class Linear(nnModule):

    def __init__(self, dim_in:int, dim_out:int):
        super().__init__()
        # Initialize weights and biases
        stdv = 1/math.sqrt(dim_in)
        self.weight = self.addParameter(Tensor(
                        np.random.uniform(-stdv, stdv, (dim_in, dim_out)),
                        requires_grad=True
                        ))
        self.bias = self.addParameter(Tensor(
                        np.random.uniform(-stdv, stdv, (dim_out, )),
                        requires_grad=True
                        ))
    
    def forward(self, input: Tensor) -> Tensor:
        """ Input dim: (*, dim_in)
            Output dim: (*, dim_out) """
        output = input.matmul(self.weight) + self.bias
        return output
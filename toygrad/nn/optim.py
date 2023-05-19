"""
optim.py
"""

# Load packages
import numpy as np
from toygrad.engine import Tensor

# - ADAM -
class ADAM:

    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """ Caution: the optimizer must be initialized each time the model is trained.
            This is because of the t_global variable is define in this way. """
        # Store values
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # Global iteration time
        self.t_global = 0
        # Initialize velocities and momentums
        self.momentums, self.velocities = [], []
        for p in self.parameters:
            self.momentums += [np.zeros(p.item.shape)]
            self.velocities += [np.zeros(p.item.shape)]
    
    def step(self):
        iter = zip(self.parameters, self.momentums, self.velocities)
        b1, b2 = self.betas
        # Update global time
        self.t_global += 1
        for ind, (p, m, v) in enumerate(iter):
            g = p.grad
            # Weight decay
            if self.weight_decay != 0:
                g = g + self.weight_decay*p.item
            # Update momentums and velocities
            self.momentums[ind] = b1*m + (1 - b1)*g
            self.velocities[ind] = b2*v + (1 - b2)*g*g
            # Rescale them
            m_hat = self.momentums[ind]/(1 - b1**self.t_global)
            v_hat = self.velocities[ind]/(1 - b2**self.t_global)
            # Update network parameters
            self.parameters[ind].item = p.item - self.lr*m_hat/(np.sqrt(v_hat) + self.eps)
        
    def zero_grad(self):
        # Set to zero the gradient of the parameters
        for p in self.parameters:
            p.grad = 0 

# - Stochastic Gradient Descent (SGD) with Momentum -
class SGD:

    def __init__(self, parameters, lr, momentum=0, weight_decay=0):
        # Store values
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocities for momentum
        self.velocities = []
        for p in self.parameters:
            self.velocities += [np.zeros(p.item.shape)]
        
    def step(self):
        for ind, (p, v) in enumerate(zip(self.parameters, self.velocities)):
            # Apply weight_decay if required
            if self.weight_decay == 0:
                p_grad = p.grad
            else:
                p_grad = p.grad + self.weight_decay*p.item
            # Update velocity vector
            self.velocities[ind] = p_grad + self.momentum*v
            # Update network parameters
            self.parameters[ind].item = p.item - self.lr*self.velocities[ind]
    
    def zero_grad(self):
        # Set to zero the gradient of the parameters
        for p in self.parameters:
            p.grad = 0 
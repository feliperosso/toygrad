# toygrad

This repository was built with the following two objectives in mind:

- Understanding the detailed inner workings of automatic differentiation packages, such a Pytorch's autograd.
- Learning the basic concepts and most common techniques used in Reinforcement Learning (RL).

We construct our own automatic differentiation engine, built on top of the numpy tensors. We have drawn inspiration 
from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) package, which has an analogous implementation
but for scalars. Using the toygrad engine, we construct an RL policy which can be trained 
using several policy gradient algorithms, including [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf) (PPO). 

We should stress no specialized neural network library such as Pytorch is used in the building of this repository. Everything is basically built from scratch using numpy tensors. Two demos are included to show what is possible with this package:

1. [demo_1](https://github.com/feliperosso/toygrad/tree/main/demo_1): We train a simple fully connected network on the MNIST dataset.
2. [demo_2](https://github.com/feliperosso/toygrad/tree/main/demo_2): We train a policy on the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) game environment of gymnasium, using two variants of Vanilla Policy Gradient as well as PPO.

Let us now briefly summarize the most relevant content of some of the modules in this repository:

### [engine.py](https://github.com/feliperosso/toygrad/blob/main/toygrad/engine.py)

This module defines the Tensor class, which is the backbone for most of the capabilities of 
the toygrad package. It basically takes numpy tensors as input and enhances their 
functionality by granting them automatic differentiation capabilities. The following 
forward and backward operations can be implemented on any instance of the Tensor class:

- Cross Entropy
- Addition
- Slicing
- Logarithm
- Matrix Multiplication
- Elementwise Multiplication/Division
- ReLU
- Reshape
- Maxiumum/Minimum
- Sum
- Sigmoid
- Softmax

These operations are compatible with numpy's broadcasting of tensors. If an additional
operation is required for a particular purpose, it shouldn't be too difficult to implement
it by extending this module.

### [models.py](https://github.com/feliperosso/toygrad/blob/main/toygrad/nn/models.py)

This modules defines the following two models:

1. MNISTNetwork: A simple fully connected network with Dropout and ReLU activation functions which can be used to classify the MNIST dataset (see [demo_1](https://github.com/feliperosso/toygrad/tree/main/demo_1)).
2. RLPolicyGradient: Defines a policy from a fully connected network which can be trained using two variants of Vanilla Policy Gradient or PPO. It also contains a value estimation network which is initialized with the policy and used during training (see [demo_2](https://github.com/feliperosso/toygrad/tree/main/demo_2))

The layers (Linear and Dropout) as well as the optimizers (SGD and ADAM) used in this module are constructed in the [layers.py](https://github.com/feliperosso/toygrad/blob/main/toygrad/nn/layers.py) and [optim.py](https://github.com/feliperosso/toygrad/blob/main/toygrad/nn/optim.py) modules.


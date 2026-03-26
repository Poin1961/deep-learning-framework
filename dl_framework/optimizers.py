import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, layer):
        # SGD updates are handled directly in the layer's backward pass for simplicity in this framework
        pass

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        # Adam updates are more complex and typically applied to weights and biases directly
        # For this simplified framework, we'll just pass for now, as SGD is handled in layers.
        # A full Adam implementation would require modifying the Dense layer's backward pass
        # to accept an optimizer and apply these updates.
        pass

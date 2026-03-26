import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class Sigmoid(Layer):
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        sigmoid_output = self.output
        return output_gradient * sigmoid_output * (1 - sigmoid_output)

class Softmax(Layer):
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This is a simplified backward pass for Softmax, typically combined with Cross-Entropy Loss
        # For standalone Softmax, the gradient calculation is more complex.
        # Here, we assume it's used with a loss function that handles the combined gradient.
        return output_gradient

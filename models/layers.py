import numpy as np

class Dense:
    """Fully connected layer."""
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        """Computes weighted sum."""
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dL_dout, learning_rate):
        """Computes gradients and updates weights."""
        dW = np.dot(self.inputs.T, dL_dout)
        db = np.sum(dL_dout, axis=0, keepdims=True)

        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

        return np.dot(dL_dout, self.weights.T)
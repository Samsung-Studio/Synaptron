import numpy as np

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dL_dout):
        dL_dout[self.inputs <= 0] = 0
        return dL_dout

class Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dL_dout):
        return dL_dout * self.output * (1 - self.output)
import numpy as np
from models.layers import Dense
from models.activations import ReLU, Sigmoid
from models.losses import MeanSquaredError
from models.optimizers import SGD

class NeuralNetwork:
    def __init__(self, layers, loss_function, optimizer):
        """
        Initialize the neural network with given layers, loss function, and optimizer.
        :param layers: List of layers (Dense + Activation)
        :param loss_function: Loss function (MSE, CrossEntropy, etc.)
        :param optimizer: Optimizer (SGD, Adam, etc.)
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, X):
        """
        Forward pass through the network.
        :param X: Input data
        :return: Final output
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dL):
        """
        Backward pass (backpropagation).
        :param dL: Gradient of loss with respect to output
        """
        for layer in reversed(self.layers):
            dL = layer.backward(dL, self.optimizer.learning_rate)

    def train(self, X_train, y_train, epochs=100):
        """
        Train the network with given data.
        :param X_train: Training features
        :param y_train: Training labels
        :param epochs: Number of epochs
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train)

            # Compute loss
            loss = self.loss_function.forward(y_pred, y_train)

            # Backward pass
            dL = self.loss_function.backward(y_pred, y_train)
            self.backward(dL)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.
        :param X: Input data
        :return: Model predictions
        """
        return self.forward(X)

if __name__ == "__main__":
    # Example: Create a simple NN and train it
    X_train = np.random.rand(100, 4)  # 100 samples, 4 features
    y_train = np.random.randint(0, 2, size=(100, 1))  # Binary labels

    # Define model architecture
    layers = [
        Dense(4, 8), ReLU(),
        Dense(8, 1), Sigmoid()
    ]

    # Create Neural Network
    model = NeuralNetwork(layers, MeanSquaredError(), SGD(learning_rate=0.01))

    # Train the model
    model.train(X_train, y_train, epochs=50)

    # Test prediction
    X_test = np.random.rand(5, 4)
    print("Predictions:", model.predict(X_test))
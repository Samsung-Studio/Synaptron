import numpy as np
from models.layers import Dense
from models.activations import ReLU
from models.losses import MeanSquaredError
from models.optimizers import SGD
from data.preprocess import preprocess_data, load_data

# Load and preprocess data
df = load_data("data/train/sample_data.csv")  # Replace with actual data
X_train, X_test, y_train, y_test = preprocess_data(df, target_column="label")

# Define model architecture
layer1 = Dense(input_size=X_train.shape[1], output_size=16)
activation1 = ReLU()
layer2 = Dense(input_size=16, output_size=1)  # Binary classification

loss_function = MeanSquaredError()
optimizer = SGD(learning_rate=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    out1 = layer1.forward(X_train)
    act1 = activation1.forward(out1)
    out2 = layer2.forward(act1)

    # Compute loss
    loss = loss_function.forward(out2, y_train)

    # Backpropagation
    dL = loss_function.backward(out2, y_train)
    dL = layer2.backward(dL, learning_rate=0.01)
    dL = activation1.backward(dL)
    layer1.backward(dL, learning_rate=0.01)

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Training Complete !")
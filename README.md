# Synaptron - Neural Network from Scratch 🧠⚡

Synaptron is a lightweight and customizable **neural network framework** built from scratch using **pure Python + NumPy**. It is designed for **educational and experimental** deep learning applications.

## 🚀 Features

- **Fully Connected Layers** (Dense)
- **Activation Functions** (ReLU, Sigmoid)
- **Loss Functions** (Mean Squared Error)
- **Gradient Descent Optimizer** (SGD)
- **Data Preprocessing & Normalization**
- **Experiment Scripts for Training Models**
- **Visualization Tools for Loss & Accuracy**

## 📂 Project Structure

```
Synaptron/
│── data/                  # Dataset storage & preprocessing
│   ├── train/             # Training data
│   ├── test/              # Testing data
│   ├── preprocess.py      # Data preprocessing script
│
│── models/                # Model implementations
│   ├── layers.py          # Implementation of Dense layers
│   ├── neural_net.py      # Neural Network class (Forward & Backpropagation)
│   ├── activations.py     # Activation functions (ReLU, Sigmoid, etc.)
│   ├── losses.py          # Loss functions (MSE, CrossEntropy)
│   ├── optimizers.py      # Optimizers (SGD, Adam, etc.)
│
│── utils/                 # Helper functions
│   ├── metrics.py         # Accuracy, Precision, Recall calculations
│   ├── visualizations.py  # Plotting loss/accuracy graphs
│   ├── helpers.py         # Miscellaneous utilities
│
│── experiments/           # Training & Experiment scripts
│   ├── experiment_1.py    # Experiment with different hyperparameters
│   ├── experiment_2.py
│
│── notebooks/             # Jupyter Notebooks for exploration
│   ├── data_analysis.ipynb
│   ├── model_testing.ipynb
│
│── main.py                # Entry point for training and evaluation
│── requirements.txt       # Required dependencies
│── README.md              # Project documentation
```

## 📊 Installation & Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Synaptron.git
   cd Synaptron
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train a model:
   ```sh
   python experiments/experiment_1.py
   ```

## 📈 Training a Neural Network

To train a simple neural network with Synaptron:

```python
from models.neural_net import NeuralNetwork
from models.layers import Dense
from models.activations import ReLU, Sigmoid
from models.losses import MeanSquaredError
from models.optimizers import SGD
import numpy as np

# Generate synthetic data
X_train = np.random.rand(100, 4)
y_train = np.random.randint(0, 2, size=(100, 1))

# Define model architecture
layers = [
    Dense(4, 8), ReLU(),
    Dense(8, 1), Sigmoid()
]

# Create and train the model
model = NeuralNetwork(layers, MeanSquaredError(), SGD(learning_rate=0.01))
model.train(X_train, y_train, epochs=50)
```

## 🔧 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 🛠️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---
# Deep Learning Framework

A lightweight and extensible deep learning framework built from scratch using Python and NumPy. This project aims to provide a clear understanding of the underlying mechanics of neural networks, backpropagation, and optimization algorithms.

## Features

*   **Modular Design**: Easily extendable with new layers, activation functions, and optimizers.
*   **Automatic Differentiation**: Implemented using a computational graph for efficient gradient calculation.
*   **GPU Acceleration (Planned)**: Future integration with CUDA for performance-critical operations.
*   **Pre-built Layers**: Includes common layers like Dense, ReLU, Sigmoid, Softmax.
*   **Loss Functions**: Supports Mean Squared Error (MSE) and Categorical Cross-Entropy.
*   **Optimizers**: Implements Stochastic Gradient Descent (SGD) and Adam.

## Installation

```bash
pip install numpy
```

## Usage

```python
import numpy as np
from dl_framework.layers import Dense, ReLU
from dl_framework.models import Sequential
from dl_framework.losses import MeanSquaredError
from dl_framework.optimizers import SGD

# 1. Prepare Data
X = np.random.rand(100, 10) # 100 samples, 10 features
y = np.random.rand(100, 1)  # 100 samples, 1 output

# 2. Build Model
model = Sequential([
    Dense(input_dim=10, output_dim=32),
    ReLU(),
    Dense(input_dim=32, output_dim=1)
])

# 3. Compile Model
model.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=0.01))

# 4. Train Model
model.fit(X, y, epochs=100, batch_size=32)

# 5. Make Predictions
predictions = model.predict(X)
print(predictions[:5])
```

## Project Structure

```
deep-learning-framework/
├── dl_framework/
│   ├── __init__.py
│   ├── layers.py
│   ├── models.py
│   ├── losses.py
│   └── optimizers.py
├── examples/
│   └── simple_regression.py
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

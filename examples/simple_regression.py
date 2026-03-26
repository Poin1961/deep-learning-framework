import numpy as np
from dl_framework.layers import Dense, ReLU
from dl_framework.models import Sequential
from dl_framework.losses import MeanSquaredError
from dl_framework.optimizers import SGD

# Generate some synthetic data for a simple regression problem
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2 * X + 1 + np.random.randn(100, 1) * 2 # y = 2x + 1 + noise

# Build the model
model = Sequential([
    Dense(input_dim=1, output_dim=16),
    ReLU(),
    Dense(input_dim=16, output_dim=1)
])

# Compile the model with Mean Squared Error loss and SGD optimizer
model.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=0.001))

# Train the model
print("\nStarting training...")
model.fit(X, y, epochs=500, batch_size=16)
print("Training finished.\n")

# Make predictions on the training data
predictions = model.predict(X)

# Evaluate the model
final_loss = MeanSquaredError().forward(predictions, y)
print(f"Final Mean Squared Error: {final_loss:.4f}")

# Print some actual vs. predicted values
print("\nActual vs. Predicted (first 10 samples):")
for i in range(10):
    print(f"Actual: {y[i][0]:.2f}, Predicted: {predictions[i][0]:.2f}")

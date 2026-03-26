from .layers import Layer

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.optimizer = None

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def fit(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            # Shuffle data for each epoch
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                output = X_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss and initial gradient
                loss_value = self.loss.forward(output, y_batch)
                gradient = self.loss.backward(output, y_batch)

                # Backward pass
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.optimizer.learning_rate)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value:.4f}")

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

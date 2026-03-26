import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = y_pred.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.mean(np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        samples = y_pred.shape[0]
        labels = y_true.shape[1]

        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((samples, labels))
            y_true_one_hot[range(samples), y_true] = 1
        else:
            y_true_one_hot = y_true

        gradient = -(y_true_one_hot / y_pred)
        gradient = gradient / samples
        return gradient

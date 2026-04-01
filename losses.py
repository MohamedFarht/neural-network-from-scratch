"""Loss functions and their derivatives."""

import numpy as np


class Loss:
    """Static methods for loss functions and their derivatives."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)

    @staticmethod
    def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def categorical_crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -y_true / (y_pred + epsilon)

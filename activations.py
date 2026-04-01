"""Activation functions and their derivatives."""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities along axis 1."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -500, 500))
    probs = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)
    return probs / np.sum(probs, axis=1, keepdims=True)


class Activation:
    """Static methods for activation functions and their derivatives."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

"""
Neural network engine — forward pass, backpropagation, training loop.

Supports single-layer and multi-layer (1–3 hidden layers) architectures
with SGD or momentum optimizers.
"""

import warnings
from typing import List, Union

import numpy as np

from activations import Activation, softmax
from losses import Loss


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Union[List[int], int, None],
        output_size: int,
        training_method: str = "Single layer",
        hidden_activations: Union[List[str], str] = "sigmoid",
        output_activation: str = "sigmoid",
        use_softmax: bool = False,
        loss_function: str = "mse",
        learning_rate: float = 0.03,
        optimizer: str = "momentum",
        momentum: float = 0.95,
        l2_lambda: float = 0.0,
    ):
        self.is_multi = training_method == "Multi layer"
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else []
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.use_softmax = use_softmax
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.momentum = momentum

        self.activation_functions = {
            "sigmoid": (Activation.sigmoid, Activation.sigmoid_derivative),
            "relu": (Activation.relu, Activation.relu_derivative),
            "tanh": (Activation.tanh, Activation.tanh_derivative),
        }

        if isinstance(hidden_activations, str):
            hidden_activations = [hidden_activations]
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation

        if self.is_multi:
            if not hidden_sizes or not 1 <= len(hidden_sizes) <= 3:
                raise ValueError("Multi-layer network must have 1-3 hidden layers")
            if any(size < 1 for size in hidden_sizes):
                raise ValueError("Hidden layer sizes must be positive")
            if len(hidden_activations) != len(hidden_sizes):
                raise ValueError(
                    "Number of activation functions must match number of hidden layers"
                )

        self.initialize_network()
        self.error_history = []
        self.validation_error_history = []

    def initialize_network(self) -> None:
        self.weights = []
        self.biases = []

        if self.optimizer == "momentum":
            self.v_weights = []
            self.v_biases = []

        if self.is_multi:
            layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
            for i in range(len(layer_sizes) - 1):
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                w = np.random.normal(0, scale, (layer_sizes[i + 1], layer_sizes[i]))
                b = np.zeros(layer_sizes[i + 1])
                self.weights.append(w)
                self.biases.append(b)
                if self.optimizer == "momentum":
                    self.v_weights.append(np.zeros_like(w))
                    self.v_biases.append(np.zeros_like(b))
        else:
            scale = np.sqrt(2.0 / (self.input_size + self.output_size))
            w = np.random.normal(0, scale, (self.output_size, self.input_size))
            b = np.zeros(self.output_size)
            self.weights = [w]
            self.biases = [b]
            if self.optimizer == "momentum":
                self.v_weights = [np.zeros_like(w)]
                self.v_biases = [np.zeros_like(b)]

        self.store_best_weights()

    def reset_optimizer_state(self):
        """Reset velocity buffers when switching optimizers."""
        if hasattr(self, "v_weights"):
            del self.v_weights
        if hasattr(self, "v_biases"):
            del self.v_biases

        if self.optimizer == "momentum":
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.v_biases = [np.zeros_like(b) for b in self.biases]

    def update_weights(self, weight_gradients, bias_gradients):
        if self.optimizer == "momentum":
            for i in range(len(self.weights)):
                self.v_weights[i] = (
                    self.momentum * self.v_weights[i]
                    + self.learning_rate * weight_gradients[i]
                )
                self.v_biases[i] = (
                    self.momentum * self.v_biases[i]
                    + self.learning_rate * bias_gradients[i]
                )
                self.weights[i] -= self.v_weights[i]
                self.biases[i] -= self.v_biases[i]
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
                self.biases[i] -= self.learning_rate * bias_gradients[i]

    def forward(self, x: np.ndarray):
        """Return (output, layer_outputs, layer_inputs).

        For single-layer networks, layer_outputs is None.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self.is_multi:
            layer_outputs = []
            layer_inputs = []
            current_input = x

            for i in range(len(self.weights) - 1):
                net = np.dot(current_input, self.weights[i].T) + self.biases[i]
                layer_inputs.append(net)
                activation_func = self.activation_functions[self.hidden_activations[i]][0]
                current_input = activation_func(net)
                layer_outputs.append(current_input)

            final_net = np.dot(current_input, self.weights[-1].T) + self.biases[-1]
            layer_inputs.append(final_net)

            if self.use_softmax:
                final_output = softmax(final_net)
            else:
                output_func = self.activation_functions[self.output_activation][0]
                final_output = output_func(final_net)

            return final_output, layer_outputs, layer_inputs
        else:
            net = np.dot(x, self.weights[0].T) + self.biases[0]
            activation_func = self.activation_functions[self.output_activation][0]
            return activation_func(net), None, [net]

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        batch_size = len(y_true)

        if self.loss_function == "mse":
            loss = Loss.mse(y_true, y_pred)
        elif self.loss_function == "binary_crossentropy":
            loss = Loss.binary_crossentropy(y_true, y_pred)
        elif self.loss_function == "categorical_crossentropy":
            loss = Loss.categorical_crossentropy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

        if self.l2_lambda > 0:
            l2_term = sum(np.sum(w ** 2) for w in self.weights)
            loss += (self.l2_lambda / (2 * batch_size)) * l2_term

        return loss

    def backward(self, X: np.ndarray, y: np.ndarray, outputs) -> float:
        batch_size = len(X)

        if self.is_multi:
            final_output, layer_outputs, layer_inputs = outputs
            layer_inputs_with_x = [X] + layer_outputs

            if self.use_softmax:
                delta = final_output - y
            else:
                if self.loss_function == "mse":
                    delta = Loss.mse_derivative(y, final_output)
                elif self.loss_function == "binary_crossentropy":
                    delta = Loss.binary_crossentropy_derivative(y, final_output)
                elif self.loss_function == "categorical_crossentropy":
                    delta = Loss.categorical_crossentropy_derivative(y, final_output)

                output_derivative = self.activation_functions[self.output_activation][1]
                delta *= output_derivative(layer_inputs[-1])

            weight_gradients = [np.zeros_like(w) for w in self.weights]
            bias_gradients = [np.zeros_like(b) for b in self.biases]

            weight_gradients[-1] = np.dot(delta.T, layer_outputs[-1]) / batch_size
            bias_gradients[-1] = np.mean(delta, axis=0)

            for i in range(len(self.weights) - 2, -1, -1):
                error = np.dot(delta, self.weights[i + 1])
                activation_derivative = self.activation_functions[self.hidden_activations[i]][1]
                delta = error * activation_derivative(layer_inputs[i])
                weight_gradients[i] = np.dot(delta.T, layer_inputs_with_x[i]) / batch_size
                bias_gradients[i] = np.mean(delta, axis=0)

        else:
            final_output, _, layer_inputs = outputs

            if self.loss_function == "mse":
                delta = Loss.mse_derivative(y, final_output)
            elif self.loss_function == "binary_crossentropy":
                delta = Loss.binary_crossentropy_derivative(y, final_output)
            elif self.loss_function == "categorical_crossentropy":
                if self.use_softmax:
                    delta = final_output - y
                else:
                    delta = Loss.categorical_crossentropy_derivative(y, final_output)

            if not self.use_softmax:
                activation_derivative = self.activation_functions[self.output_activation][1]
                delta *= activation_derivative(layer_inputs[0])

            weight_gradients = [np.dot(delta.T, X) / batch_size]
            bias_gradients = [np.mean(delta, axis=0)]

        if self.l2_lambda > 0:
            for i in range(len(self.weights)):
                weight_gradients[i] += (self.l2_lambda / batch_size) * self.weights[i]

        self.update_weights(weight_gradients, bias_gradients)
        return self.compute_loss(y, final_output)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data=None,
        max_epochs: int = 1000,
        target_error: float = 0.01,
        batch_size: int = 32,
        patience: int = 10,
        min_learning_rate: float = 0.0001,
    ):
        num_samples = len(X)
        self.error_history = []
        self.validation_error_history = []

        best_error = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(max_epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_error = 0
            num_batches = (num_samples + batch_size - 1) // batch_size

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                outputs = self.forward(X_batch)
                batch_error = self.backward(X_batch, y_batch, outputs)
                total_error += batch_error

            avg_error = total_error / num_batches
            self.error_history.append(avg_error)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_outputs = self.forward(X_val)
                val_error = self.compute_loss(y_val, val_outputs[0])
                self.validation_error_history.append(val_error)
                current_error = val_error
            else:
                current_error = avg_error

            if current_error < best_error:
                best_error = current_error
                best_epoch = epoch + 1
                self.store_best_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch > 0 and self.error_history[-1] > self.error_history[-2]:
                self.learning_rate = max(self.learning_rate * 0.5, min_learning_rate)
            else:
                self.learning_rate = min(
                    self.learning_rate * 1.1, self.initial_learning_rate
                )

            if (epoch + 1) % 100 == 0:
                print(f"epoch {epoch + 1}/{max_epochs}, err={avg_error:.6f}")

            if patience_counter >= patience:
                print(f"early stop @ epoch {epoch + 1}")
                break

            if avg_error <= target_error:
                print(f"target error reached @ epoch {epoch + 1}")
                break

        self.restore_best_weights()
        return best_epoch, best_error

    def store_best_weights(self) -> bool:
        if not hasattr(self, "weights") or not hasattr(self, "biases"):
            return False
        if len(self.weights) != len(self.biases):
            return False

        self.best_weights = [w.copy() for w in self.weights]
        self.best_biases = [b.copy() for b in self.biases]

        if self.optimizer == "momentum" and hasattr(self, "v_weights"):
            self.best_v_weights = [v.copy() for v in self.v_weights]
            self.best_v_biases = [v.copy() for v in self.v_biases]

        return True

    def restore_best_weights(self) -> bool:
        if not hasattr(self, "best_weights") or not self.best_weights:
            warnings.warn("No best weights available to restore")
            return False

        self.weights = [w.copy() for w in self.best_weights]
        self.biases = [b.copy() for b in self.best_biases]

        if self.optimizer == "momentum" and hasattr(self, "best_v_weights"):
            self.v_weights = [v.copy() for v in self.best_v_weights]
            self.v_biases = [v.copy() for v in self.best_v_biases]

        return True

    def adjust_learning_rate(self, epoch: int, decay_rate: float = 0.1) -> None:
        self.learning_rate = self.initial_learning_rate * (
            1.0 / (1.0 + decay_rate * epoch)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        outputs = self.forward(X)
        return np.argmax(outputs[0], axis=1)

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes if self.is_multi else None,
            "output_size": self.output_size,
            "training_method": "Multi layer" if self.is_multi else "Single layer",
            "hidden_activations": self.hidden_activations,
            "output_activation": self.output_activation,
            "learning_rate": self.initial_learning_rate,
            "momentum": self.momentum,
            "l2_lambda": self.l2_lambda,
        }

    def __str__(self) -> str:
        config = self.get_config()
        layers = (
            [config["input_size"]]
            + (config["hidden_sizes"] if config["hidden_sizes"] else [])
            + [config["output_size"]]
        )

        if self.use_softmax:
            activation_info = "Output: Softmax"
        else:
            activation_info = f"Output: {config['output_activation']}"

        if config["hidden_activations"]:
            activation_info += f"\nHidden: {', '.join(config['hidden_activations'])}"

        return (
            f"Neural Network Configuration:\n"
            f"Architecture: {config['training_method']}\n"
            f"Layer sizes: {layers}\n"
            f"Activations: {activation_info}\n"
            f"Learning rate: {config['learning_rate']}\n"
            f"Momentum: {config['momentum']}\n"
            f"L2 regularization: {config['l2_lambda']}"
        )

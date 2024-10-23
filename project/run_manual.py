"""
Make sure you have minitorch installed in your virtual environment.
To install it, run:
>>> pip install -Ue .
"""

import random
import minitorch
from typing import List, Tuple
from minitorch.operators import sigmoid


class Network(minitorch.Module):
    """A simple model that can learn from data."""

    def __init__(self) -> None:
        """Set up the model with one layer that connects two inputs to one output."""
        super().__init__()
        self.linear: Linear = Linear(
            2, 1
        )  # Create a layer that takes 2 inputs and gives 1 output

    def forward(self, x: Tuple[float, float]) -> float:
        """Process the input through the model and use a special function to adjust the output."""
        y = self.linear(x)  # Get the output from the layer
        return sigmoid(y[0])  # Apply a function to make the output between 0 and 1


class Linear(minitorch.Module):
    """A layer that performs a simple calculation to connect inputs and outputs."""

    def __init__(self, in_size: int, out_size: int) -> None:
        """Set up the layer with random weights and biases."""
        super().__init__()
        random.seed(100)  # Keep the randomness the same each time for consistency
        self.weights: List[List[minitorch.Parameter]] = []  # This will hold the weights
        self.bias: List[minitorch.Parameter] = []  # This will hold the biases

        # Set up weights for the connections
        for i in range(in_size):
            weights = []
            for j in range(out_size):
                w = self.add_parameter(
                    f"weight_{i}_{j}", 2 * (random.random() - 0.5)
                )  # Random weight
                weights.append(w)
            self.weights.append(weights)

        # Set up biases for the outputs
        for j in range(out_size):
            b = self.add_parameter(
                f"bias_{j}", 2 * (random.random() - 0.5)
            )  # Random bias
            self.bias.append(b)

    def forward(self, inputs: Tuple[float, float]) -> List[float]:
        """Calculate the output by applying the weights and biases to the inputs."""
        y: List[float] = [b.data for b in self.bias]  # Get the bias values
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = (
                    y[j]
                    + x
                    * self.weights[i][j].data  # Update the output with input and weight
                )
        return y  # Return the calculated output values


class ManualTrain:
    """A helper class to train the model."""

    def __init__(self, hidden_layers: int) -> None:
        """Set up the training class with a model."""
        self.model: Network = Network()  # Create an instance of the Network

    def run_one(self, x: Tuple[float, float]) -> float:
        """Run the model with a single input."""
        return self.model.forward((x[0], x[1]))  # Process the input through the model

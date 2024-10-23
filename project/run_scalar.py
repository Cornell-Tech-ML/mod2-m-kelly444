import random
from typing import Callable, List, Tuple, Any
import minitorch


class Network(minitorch.Module):
    """A model with three layers for processing data."""

    def __init__(self, hidden_layers: int) -> None:
        """Set up the model with layers. The first two layers use hidden nodes."""
        super().__init__()
        self.layer1: Linear = Linear(
            2, hidden_layers
        )  # First layer connects 2 inputs to hidden nodes
        self.layer2: Linear = Linear(
            hidden_layers, hidden_layers
        )  # Second layer connects hidden nodes to more hidden nodes
        self.layer3: Linear = Linear(
            hidden_layers, 1
        )  # Last layer connects hidden nodes to 1 output

    def forward(self, x: Tuple[minitorch.Scalar, minitorch.Scalar]) -> minitorch.Scalar:
        """Process the input through the layers and apply a function to adjust the output."""
        middle: List[minitorch.Scalar] = self.layer1.forward(
            [x[0], x[1]]
        )  # Forward through the first layer
        middle = [
            h.relu() for h in middle
        ]  # Apply activation function after first layer
        end: List[minitorch.Scalar] = self.layer2.forward(
            middle
        )  # Forward through the second layer
        end = [h.relu() for h in end]  # Apply activation after second layer
        return self.layer3.forward(end)[
            0
        ].sigmoid()  # Get the final output and squash it to be between 0 and 1


class Linear(minitorch.Module):
    """A layer that performs linear transformations using weights and biases."""

    def __init__(self, in_size: int, out_size: int) -> None:
        """Set up the layer with random weights and biases."""
        super().__init__()
        self.weights: List[List[minitorch.Parameter]] = [
            [] for _ in range(in_size)
        ]  # Initialize weights
        self.bias: List[minitorch.Parameter] = []  # This will hold biases

        # Initialize weights for connections
        for i in range(in_size):
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}",
                        minitorch.Scalar(
                            2 * (random.random() - 0.5), requires_grad=True
                        ),  # Random weight
                    )
                )

        # Initialize biases for the outputs
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}",
                    minitorch.Scalar(
                        2 * (random.random() - 0.5), requires_grad=True
                    ),  # Random bias
                )
            )

    def forward(self, inputs: List[minitorch.Scalar]) -> List[minitorch.Scalar]:
        """Calculate the output using inputs, weights, and biases."""
        output: List[minitorch.Scalar] = []
        for j in range(len(self.bias)):
            weighted_sum: float = sum(
                inputs[i].data * self.weights[i][j].data for i in range(len(inputs))
            )
            # Directly append the bias to the weighted sum
            weighted_sum += self.bias[j].data
            output.append(
                minitorch.Scalar(weighted_sum, requires_grad=True)
            )  # Wrap the output in Scalar
        return output  # Return the calculated output values


def default_log_fn(
    epoch: int, total_loss: float, correct: int, losses: List[float]
) -> None:
    """Default logging function to show training progress."""
    print("Epoch", epoch, "loss", total_loss, "correct", correct)


class ScalarTrain:
    """A helper class to train the model."""

    def __init__(self, hidden_layers: int) -> None:
        """Set up the training class with a model."""
        self.hidden_layers: int = hidden_layers
        self.model: Network = Network(self.hidden_layers)  # Create the model

    def run_one(self, x: Tuple[float, float]) -> minitorch.Scalar:
        """Run the model with a single input."""
        return self.model.forward(
            (
                minitorch.Scalar(x[0], name="x_1", requires_grad=True),  # First input
                minitorch.Scalar(x[1], name="x_2", requires_grad=True),  # Second input
            )
        )

    def train(
        self,
        data: Any,
        learning_rate: float,
        max_epochs: int = 500,
        log_fn: Callable[[int, float, int, List[float]], None] = default_log_fn,
    ) -> None:
        """Train the model using the provided data."""
        self.learning_rate: float = learning_rate
        self.max_epochs: int = max_epochs
        optim = minitorch.SGD(
            self.model.parameters(), learning_rate
        )  # Set up the optimizer

        losses: List[float] = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss: float = 0.0  # Track total loss for the epoch
            correct: int = 0  # Count correct predictions
            optim.zero_grad()  # Reset gradients

            # Loop through each data point
            for i in range(data.N):
                x_1, x_2 = data.X[i]  # Get inputs
                y = data.y[i]  # Get true output
                x_1 = minitorch.Scalar(
                    x_1, requires_grad=True
                )  # Create scalar for first input
                x_2 = minitorch.Scalar(
                    x_2, requires_grad=True
                )  # Create scalar for second input
                out = self.model.forward((x_1, x_2))  # Get the model output

                # Determine loss based on true output
                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0  # Count correct prediction
                else:
                    prob = 1.0 - out
                    correct += 1 if out.data < 0.5 else 0  # Count correct prediction

                # Calculate the loss using cross-entropy
                loss = -((y * out.log()) + ((1 - y) * prob.log()))
                total_loss += loss.data  # Accumulate loss

                # Backward pass to compute gradients
                loss.backward()

            losses.append(total_loss / data.N)  # Average loss for the epoch

            optim.step()  # Update model parameters

            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss / data.N, correct, losses)


if __name__ == "__main__":
    PTS: int = 50  # Number of data points
    HIDDEN: int = 2  # Number of hidden layers
    RATE: float = 0.5  # Learning rate
    data = minitorch.datasets["Simple"](PTS)  # Load simple dataset
    ScalarTrain(HIDDEN).train(data, RATE)  # Train the model

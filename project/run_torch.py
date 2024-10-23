import torch
import minitorch
import numpy as np
from typing import Callable, List, Any


def default_log_fn(
    epoch: int, total_loss: float, correct: int, losses: List[float]
) -> None:
    print("Epoch", epoch, "loss", total_loss, "correct", correct)


class Linear(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            2 * (torch.rand((in_size, out_size)) - 0.5)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(
            2 * (torch.rand((out_size,)) - 0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class Network(torch.nn.Module):
    def __init__(self, hidden_layers: int) -> None:
        super().__init__()
        self.layer1: Linear = Linear(2, hidden_layers)
        self.layer2: Linear = Linear(hidden_layers, hidden_layers)
        self.layer3: Linear = Linear(hidden_layers, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class TorchTrain:
    def __init__(self, hidden_layers: int) -> None:
        self.hidden_layers: int = hidden_layers
        self.model: Network = Network(hidden_layers)

    def run_one(self, x: float) -> torch.Tensor:
        return self.model.forward(torch.tensor([x], dtype=torch.float32))

    def run_many(self, X: List[float]) -> torch.Tensor:
        return self.model.forward(torch.tensor(X, dtype=torch.float32)).detach()

    def train(
        self,
        data: Any,
        learning_rate: float,
        max_epochs: int = 500,
        log_fn: Callable[[int, float, int, List[float]], None] = default_log_fn,
    ) -> None:
        # Prepare data tensors explicitly
        X_tensor = torch.tensor(data.X, dtype=torch.float32)
        y_tensor = torch.tensor(data.y, dtype=torch.float32).view(-1)

        self.model = Network(self.hidden_layers)
        self.max_epochs: int = max_epochs
        model: Network = self.model

        losses: List[float] = []
        for epoch in range(1, max_epochs + 1):
            # Forward
            out: torch.Tensor = model.forward(X_tensor).view(data.N)
            probs: torch.Tensor = (out * y_tensor) + (out - 1.0) * (y_tensor - 1.0)
            loss: torch.Tensor = -probs.log().sum()

            # Update
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    p.data -= learning_rate * (p.grad / float(data.N))
                    p.grad.zero_()

            # Convert tensors to NumPy for correct prediction counting
            pred_np = (out.detach().numpy() > 0.5).astype(
                int
            )  # Binary predictions as int
            y_np = y_tensor.numpy().astype(int)  # Convert y_tensor to int

            # Calculate correct predictions using NumPy
            correct = np.sum(pred_np == y_np)  # Sum the correct predictions
            loss_num: float = loss.item()
            losses.append(loss_num)

            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, loss_num, correct, losses)


if __name__ == "__main__":
    PTS: int = 250
    HIDDEN: int = 10
    RATE: float = 0.5
    TorchTrain(HIDDEN).train(minitorch.datasets["Xor"](PTS), RATE)

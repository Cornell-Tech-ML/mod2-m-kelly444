import minitorch
from typing import Callable, List, Any


# Use this function to make a random parameter in
# your module.
def RParam(*shape: int) -> minitorch.Parameter:
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


def default_log_fn(
    epoch: int, total_loss: float, correct: int, losses: List[float]
) -> None:
    print("Epoch ", epoch, " loss ", total_loss, " correct ", correct)


class Network:
    def __init__(self, hidden_layers: int):
        self.layers = []
        input_size = 2  # Adjust according to your input size
        for _ in range(hidden_layers):
            self.layers.append(minitorch.Linear(input_size, input_size))
            # Adjust input size if you want different sizes for layers
        self.layers.append(minitorch.Linear(input_size, 1))  # Output layer

    def forward(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer(x).relu()
        return x


class TensorTrain:
    def __init__(self, hidden_layers: int) -> None:
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x: Any) -> Any:
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X: List[Any]) -> Any:
        return self.model.forward(minitorch.tensor(X))

    def train(
        self,
        data: Any,
        learning_rate: float,
        max_epochs: int = 500,
        log_fn: Callable[[int, float, int, List[float]], None] = default_log_fn,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses: List[float] = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

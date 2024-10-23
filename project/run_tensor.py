import minitorch
from typing import Callable, List, Any


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.weights = minitorch.rand((in_features, out_features))
        self.bias = minitorch.rand((out_features,))

    def __call__(self, x: Any) -> Any:
        return x @ self.weights + self.bias


def RParam(*shape: int) -> minitorch.Parameter:
    r = minitorch.tensor(2.0) * (minitorch.rand(shape) - minitorch.tensor(0.5))
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
            self.layers.append(Linear(input_size, input_size))
        self.layers.append(Linear(input_size, 1))

    def forward(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer(x).relu()
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]


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
            prob = (out * y) + (out - minitorch.tensor(1.0)) * (
                y - minitorch.tensor(1.0)
            )

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

from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for optimization algorithms.

    Attributes
    ----------
        parameters: A sequence of parameters to be optimized.

    """

    def __init__(self, parameters: Sequence[Parameter]):
        """Initializes the Optimizer with a sequence of parameters.

        Args:
        ----
            parameters: A sequence of `Parameter` instances to optimize.

        """
        self.parameters = parameters


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    Attributes
    ----------
        lr: Learning rate for the optimization process.

    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Initializes the SGD optimizer.

        Args:
        ----
            parameters: A sequence of `Parameter` instances to optimize.
            lr: Learning rate for the optimizer. Defaults to 1.0.

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Resets the gradients of all parameters to zero.

        This method ensures that gradients are cleared before the next
        optimization step, preventing accumulation from previous steps.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Updates the parameters based on the gradients.

        This method performs a single optimization step, adjusting the
        parameters in the direction of the negative gradient scaled by
        the learning rate.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)

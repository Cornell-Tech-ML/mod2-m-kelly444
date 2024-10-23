from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union
import numpy as np
from .autodiff import Context, Variable, backpropagate
from .scalar_functions import (
    Add,
    Subtract,
    Multiply,
    Divide,
    ReLU as ReLUFunction,
    ScalarFunction,
)

ScalarLike = Union[float, int, "Scalar"]


def to_float(value: ScalarLike) -> float:
    """Convert a ScalarLike object to a float."""
    if isinstance(value, Scalar):
        return value.data
    return float(value)


@dataclass
class ScalarHistory:
    """Class to hold the history of operations for a Scalar."""

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Any] = ()

    def requires_grad(self) -> bool:
        """Check if any inputs require gradients."""
        return any(
            getattr(input_scalar, "requires_grad", False)
            for input_scalar in self.inputs
        )


_var_count = 0


@dataclass
class Scalar:
    """Class representing a scalar value with support for automatic differentiation."""

    data: float
    requires_grad: bool = False
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    grad: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data}, requires_grad={self.requires_grad})"

    def __hash__(self) -> int:
        return hash(self.unique_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scalar):
            return NotImplemented
        return self.unique_id == other.unique_id and self.data == other.data

    def set_history(
        self,
        last_fn: Type[ScalarFunction],
        ctx: Optional[Context] = None,
        inputs: Sequence[Scalar] = (),
    ) -> None:
        """Set the history of the scalar for gradient tracking."""
        if self.requires_grad or any(
            input_scalar.requires_grad for input_scalar in inputs
        ):
            self.history = ScalarHistory(last_fn=last_fn, ctx=ctx, inputs=inputs)

    def _get_ctx(self, ctx: Optional[Context]) -> Context:
        """Get the context for the operation."""
        return ctx if ctx is not None else Context()

    def is_leaf(self) -> bool:
        """Check if the scalar is a leaf node in the computation graph."""
        return self.history is None or self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the scalar is a constant."""
        return self.history is None

    def __add__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Add another scalar or scalar-like value."""
        ctx = self._get_ctx(ctx)
        result = Scalar(
            to_float(self.data) + to_float(b),
            requires_grad=self.requires_grad
            or (isinstance(b, Scalar) and b.requires_grad),
        )
        result.set_history(
            Add,
            ctx=ctx,
            inputs=[self]
            + ([b] if isinstance(b, Scalar) else [Scalar(b, requires_grad=True)]),
        )
        return result

    def __radd__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Right add operation."""
        return self.__add__(b, ctx)

    def __sub__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Subtract another scalar or scalar-like value."""
        ctx = self._get_ctx(ctx)
        result = Scalar(
            to_float(self.data) - to_float(b),
            requires_grad=self.requires_grad
            or (isinstance(b, Scalar) and b.requires_grad),
        )
        result.set_history(
            Subtract,
            ctx=ctx,
            inputs=[self]
            + ([b] if isinstance(b, Scalar) else [Scalar(b, requires_grad=True)]),
        )
        return result

    def __rsub__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Right subtract operation."""
        return Scalar(to_float(b) - to_float(self.data), requires_grad=True)

    def __mul__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Multiply by another scalar or scalar-like value."""
        ctx = self._get_ctx(ctx)
        result = Scalar(
            to_float(self.data) * to_float(b),
            requires_grad=self.requires_grad
            or (isinstance(b, Scalar) and b.requires_grad),
        )
        result.set_history(
            Multiply,
            ctx=ctx,
            inputs=[self]
            + ([b] if isinstance(b, Scalar) else [Scalar(b, requires_grad=True)]),
        )
        return result

    def __rmul__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Right multiply operation."""
        return self.__mul__(b, ctx)

    def __truediv__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Divide by another scalar or scalar-like value."""
        ctx = self._get_ctx(ctx)
        if to_float(b) == 0:
            raise ValueError("Division by zero is not allowed.")
        result = Scalar(
            to_float(self.data) / to_float(b),
            requires_grad=self.requires_grad
            or (isinstance(b, Scalar) and b.requires_grad),
        )
        result.set_history(
            Divide,
            ctx=ctx,
            inputs=[self]
            + ([b] if isinstance(b, Scalar) else [Scalar(b, requires_grad=True)]),
        )
        return result

    def __rtruediv__(self, b: ScalarLike, ctx: Optional[Context] = None) -> Scalar:
        """Right division operation."""
        if to_float(self.data) == 0:
            raise ValueError("Division by zero is not allowed.")
        return Scalar(to_float(b) / to_float(self.data), requires_grad=True)

    def __neg__(self) -> Scalar:
        """Negate the scalar value."""
        return Scalar(-self.data, requires_grad=self.requires_grad)

    def __lt__(self, b: ScalarLike) -> bool:
        """Check if the scalar is less than another value."""
        return self.data < to_float(b)

    def __gt__(self, b: ScalarLike) -> bool:
        """Check if the scalar is greater than another value."""
        return self.data > to_float(b)

    def relu(self) -> Scalar:
        """Apply the ReLU activation function."""
        return ReLUFunction.apply(self)

    def sigmoid(self) -> Scalar:
        """Apply the sigmoid activation function."""
        return Scalar(1 / (1 + np.exp(-self.data)), requires_grad=self.requires_grad)

    def log(self) -> Scalar:
        """Compute the natural logarithm of the scalar."""
        if self.data <= 0:
            raise ValueError("Logarithm is only defined for positive values.")
        return Scalar(np.log(self.data), requires_grad=self.requires_grad)

    def exp(self) -> Scalar:
        """Compute the exponential of the scalar."""
        result = Scalar(np.exp(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.set_history(
                ScalarFunction,  # Use the appropriate function type
                ctx=self._get_ctx(None),
                inputs=[self],
            )
        return result

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative during backpropagation."""
        if not self.is_leaf():
            return
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of this scalar."""
        assert self.history is not None, "This scalar has no history."
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for backpropagation."""
        h = self.history
        if h is None or h.last_fn is None:
            return []

        ctx = h.ctx if h.ctx is not None else Context()
        derivatives = h.last_fn._backward(ctx, d_output)
        return [(var, der) for var, der in zip(h.inputs, derivatives)]

    def backward(self, d_output: Optional[float] = None) -> None:
        """Perform backpropagation from this scalar."""
        if not self.requires_grad:
            raise Exception("This scalar does not require gradients.")

        if any(
            not isinstance(parent, Scalar) or not parent.requires_grad
            for parent in self.parents
        ):
            raise Exception("All parent Scalars must have requires_grad=True.")

        if d_output is None:
            d_output = 1.0

        backpropagate(self, d_output)


def derivative_check(f: Any, *inputs: Scalar) -> None:
    """Check the gradient of the function at given inputs."""
    for input in inputs:
        if not input.requires_grad:
            raise ValueError("All inputs must require gradients.")
    eps = 1e-5
    for i, input in enumerate(inputs):
        x = input.data
        input.data = x + eps
        f_plus = f(*inputs)
        input.data = x - eps
        f_minus = f(*inputs)
        input.data = x
        numerical_derivative = (f_plus - f_minus) / (2 * eps)
        print(f"Numerical derivative at input {i}: {numerical_derivative}")

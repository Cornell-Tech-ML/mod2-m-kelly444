from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, Tuple, Union, runtime_checkable

import minitorch
from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from .scalar import Scalar


@runtime_checkable
class Variable(Protocol):
    """Protocol defining the interface for variables."""

    @property
    def data(self) -> float:
        """Get the value stored in the variable."""
        ...


if TYPE_CHECKING:
    # A ScalarLike can be a float, int, Scalar, or Variable
    ScalarLike = Union[float, int, Scalar, Variable]


def wrap_tuple(x: Union[float, Tuple[float, ...]]) -> Tuple[float, ...]:
    """Ensure a single number or a group of numbers is in a tuple."""
    return x if isinstance(x, tuple) else (x,)


class ScalarFunction:
    """Blueprint for scalar functions with forward and backward computation."""

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Calculate input changes based on output changes (backward pass)."""
        return wrap_tuple(cls.backward(ctx, d_out))

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Calculate the output of the function using input numbers (forward pass)."""
        return cls.forward(ctx, *inps)

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Use the scalar function with given input values and return the result."""
        raw_vals = []  # Hold the actual numbers
        scalars = []  # Hold the Scalar objects

        for v in vals:
            if isinstance(v, (minitorch.scalar.Scalar, Variable)):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                # Create a Scalar object for raw numbers
                scalar_obj = minitorch.scalar.Scalar(v, requires_grad=True)
                scalars.append(scalar_obj)
                raw_vals.append(scalar_obj.data)

        ctx = Context(False)  # Create a context for the operation
        c = cls._forward(ctx, *raw_vals)  # Get the output value

        # Ensure the output is a number
        assert isinstance(c, (float, int)), f"Expected a number, got {type(c)}"

        # Save the history for later use
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(float(c), requires_grad=True, history=back)

    @staticmethod
    def forward(ctx: Context, *inps: float) -> float:
        """Actual function calculation. Needs to be defined in a subclass."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """How inputs should change based on output change. Needs to be defined in a subclass."""
        raise NotImplementedError


class Add(ScalarFunction):
    """Function to add two numbers together: $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the sum of two numbers."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The output change applies equally to both inputs."""
        return d_output, d_output


class Subtract(ScalarFunction):
    """Function to subtract one number from another: $f(x, y) = x - y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the difference between two numbers."""
        return a - b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The change in output affects the first number positively and the second negatively."""
        return d_output, -d_output


class Multiply(ScalarFunction):
    """Function to multiply two numbers: $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the product of two numbers."""
        ctx.save_for_backward(a, b)  # Remember inputs for later
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The output change depends on both inputs."""
        a, b = ctx.saved_values  # Get the saved inputs
        return d_output * b, d_output * a


class Divide(ScalarFunction):
    """Function to divide one number by another: $f(x, y) = x / y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the quotient of two numbers."""
        ctx.save_for_backward(a, b)  # Remember inputs for later
        return a / b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The change in output is affected by both inputs."""
        a, b = ctx.saved_values  # Get the saved inputs
        return d_output / b, -d_output * a / (b**2)


class Neg(ScalarFunction):
    """Function to negate a number: $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Flip the sign of the number."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Change in output flips its sign."""
        return -d_output


class Inv(ScalarFunction):
    """Function to calculate the inverse: $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the inverse of a number."""
        ctx.save_for_backward(a)  # Remember the input
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Change in output depends on the square of the input value."""
        (a,) = ctx.saved_values  # Get the saved input
        return -d_output / (a**2)


class Log(ScalarFunction):
    """Function to calculate the logarithm: $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the logarithm of a number."""
        ctx.save_for_backward(a)  # Remember the input
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Change in output is determined by the logarithm of the input."""
        (a,) = ctx.saved_values  # Get the saved input
        return operators.log_back(a, d_output)


class Exp(ScalarFunction):
    """Function to calculate the exponential: $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the exponential of a number."""
        ctx.save_for_backward(a)  # Remember the input
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Change in output is based on the input's exponential value."""
        (a,) = ctx.saved_values  # Get the saved input
        return d_output * operators.exp(a)


class ReLU(ScalarFunction):
    """Function to apply the ReLU (Rectified Linear Unit): $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """If the number is positive, keep it; else return zero."""
        ctx.save_for_backward(a)  # Remember the input
        return max(0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """If the input was positive, pass through the change; else contribute nothing."""
        (a,) = ctx.saved_values  # Get the saved input
        return d_output if a > 0 else 0


class Sigmoid(ScalarFunction):
    """Function to calculate the sigmoid: $f(x) = 1/(1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the sigmoid value, squashing inputs between 0 and 1."""
        exp_neg_a = operators.exp(-a)
        ctx.save_for_backward(exp_neg_a)  # Remember the intermediate value
        return 1 / (1 + exp_neg_a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate how the input should change based on output change using the sigmoid formula."""
        exp_neg_a = ctx.saved_values[0]  # Get the saved intermediate value
        return d_output * exp_neg_a * (1 - exp_neg_a)  # Derivative of the sigmoid


class LT(ScalarFunction):
    """Function to check if one number is less than another: $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1 (true) if a is less than b, otherwise return 0 (false)."""
        return float(a < b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The comparison doesn't affect how inputs change, return zero changes for both."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Function to check if two numbers are equal: $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1 (true) if a equals b, otherwise return 0 (false)."""
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The equality comparison doesn't affect how inputs change, return zero changes for both."""
        return 0.0, 0.0

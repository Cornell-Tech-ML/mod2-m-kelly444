from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union

import minitorch
from . import operators
from .autodiff import Context, Variable

if TYPE_CHECKING:
    from .scalar import Scalar

    # A ScalarLike can be a float, int, Scalar, or MyVariable
    ScalarLike = Union[float, int, Scalar, Variable]


def wrap_tuple(x: Union[float, Tuple[float, ...]]) -> Tuple[float, ...]:
    """Take a single number or a group of numbers and make sure it's in a tuple."""
    return x if isinstance(x, tuple) else (x,)


class ScalarFunction:
    """A general blueprint for functions that work with single numbers (scalars) and have methods for calculating their outputs in two steps: forward and backward."""

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Calculate how to change the input numbers based on how the output changed (backward pass)."""
        return wrap_tuple(cls.backward(ctx, d_out))

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Calculate the output of the function using the input numbers (forward pass)."""
        return cls.forward(ctx, *inps)

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Use the scalar function with given input values and return the result."""
        raw_vals = []  # This will hold the actual numbers
        scalars = []  # This will hold the Scalar objects

        for v in vals:
            if isinstance(v, (minitorch.scalar.Scalar, Variable)):
                scalars.append(v)
                raw_vals.append(
                    v.value if isinstance(v, Variable) else v.data
                )  # Get the number from MyVariable or Scalar
            else:
                # If it's just a number, make it a Scalar object
                scalar_obj = minitorch.scalar.Scalar(v, requires_grad=True)
                scalars.append(scalar_obj)
                raw_vals.append(scalar_obj.data)

        ctx = Context(False)  # Create a context to keep track of what's happening
        c = cls._forward(ctx, *raw_vals)  # Get the output value

        # Make sure the output is a number
        assert isinstance(c, (float, int)), f"Expected a number, got {type(c)}"

        # Save the history of the calculation for later use
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(float(c), requires_grad=True, history=back)

    @staticmethod
    def forward(ctx: Context, *inps: float) -> float:
        """The actual calculation for the function. This needs to be defined in a specific function (subclass)."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """How the inputs should change based on the change in output. This needs to be defined in a specific function (subclass)."""
        raise NotImplementedError


# Here we define specific operations like addition, subtraction, etc.


class Add(ScalarFunction):
    """Function to add two numbers together: $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the sum of two numbers."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """When we know how much the output changed, say the same change applies to both inputs (they both contribute equally)."""
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
        ctx.save_for_backward(a, b)  # Remember the inputs for later
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The change in output depends on both inputs: how much the output changes when either input changes."""
        a, b = ctx.saved_values  # Get the saved inputs
        return d_output * b, d_output * a


class Divide(ScalarFunction):
    """Function to divide one number by another: $f(x, y) = x / y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Calculate the quotient of two numbers."""
        ctx.save_for_backward(a, b)  # Remember the inputs for later
        return a / b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The change in output is affected by both inputs: how much the output changes based on the divisor and dividend."""
        a, b = ctx.saved_values  # Get the saved inputs
        return d_output / b, -d_output * a / (b**2)


class Neg(ScalarFunction):
    """Function to negate a number: $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Simply flip the sign of the number."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The change in output just flips its sign for the input."""
        return -d_output


class Inv(ScalarFunction):
    """Function to calculate the inverse: $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the inverse of a number."""
        ctx.save_for_backward(a)  # Remember the input for later
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The change in output depends on the square of the input value."""
        (a,) = ctx.saved_values  # Get the saved input
        return -d_output / (a**2)


class Log(ScalarFunction):
    """Function to calculate the logarithm: $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the logarithm of a number."""
        ctx.save_for_backward(a)  # Remember the input for later
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The change in output is determined by the logarithm of the input and the output change."""
        (a,) = ctx.saved_values  # Get the saved input
        return operators.log_back(a, d_output)


class Exp(ScalarFunction):
    """Function to calculate the exponential: $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the exponential of a number."""
        ctx.save_for_backward(a)  # Remember the input for later
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The change in output is based on the original input's exponential value."""
        (a,) = ctx.saved_values  # Get the saved input
        return d_output * operators.exp(a)


class ReLU(ScalarFunction):
    """Function to apply the ReLU (Rectified Linear Unit): $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """If the number is positive, keep it; if not, return zero."""
        ctx.save_for_backward(a)  # Remember the input for later
        return max(0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """If the input was positive, pass through the change; if not, it contributes nothing."""
        (a,) = ctx.saved_values  # Get the saved input
        return d_output if a > 0 else 0


class Sigmoid(ScalarFunction):
    """Function to calculate the sigmoid: $f(x) = 1/(1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Calculate the sigmoid value, which squashes inputs to be between 0 and 1."""
        exp_neg_a = operators.exp(-a)
        ctx.save_for_backward(exp_neg_a)  # Remember the intermediate value for backward
        return 1 / (1 + exp_neg_a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Calculate how the input should change based on the change in output using the sigmoid formula."""
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
        """The comparison doesn't affect how inputs change, so return zero changes for both inputs."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Function to check if two numbers are equal: $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return 1 (true) if a equals b, otherwise return 0 (false)."""
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The equality comparison doesn't affect how inputs change, so return zero changes for both inputs."""
        return 0.0, 0.0

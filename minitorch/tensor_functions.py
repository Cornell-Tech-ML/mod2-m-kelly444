"""Implementation of autodifferentiation Functions for Tensor operations."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend, Optional

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Convert an input value into a tuple.

    If the input is already a tuple, return it as is.
    Otherwise, wrap it in a tuple.
    """
    if isinstance(x, tuple):
        return x
    return (x,)


# Base class for all operations
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Calculate the backward pass for an operation."""
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Perform the forward pass for an operation."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Run the forward function and keep track of the operation history.

        This allows for calculating gradients later on.
        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create a context to store information needed for backpropagation.
        ctx = Context(not need_grad)

        # Execute the forward function with the input values.
        c = cls._forward(ctx, *raw_vals)

        # Prepare the backward history if needed.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negate each element in the tensor, i.e., $f(t1) = -x$."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for negation."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for negation, returning $-grad_output$."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Calculate the inverse of each element in the tensor, i.e., $f(t1) = 1/t1$."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for inversion."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for inversion."""
        (t1,) = ctx.saved_tensors
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Add two tensors element-wise, i.e., $f(t1, t2) = t1 + t2$."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for addition."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for addition."""
        return grad_output, grad_output


class All(Function):
    """Check if all elements are true. Returns 1 if all are true, otherwise 0."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Perform the forward pass for checking if all are true."""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(
                a.contiguous().view(
                    minitorch.Tensor.make([-1], (1,), backend=a.backend)
                ),
                0,
            )


class Mul(Function):
    """Multiply two tensors element-wise, i.e., $f(t1, t2) = t1 * t2$."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for multiplication."""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for multiplication."""
        (t1, t2) = ctx.saved_tensors
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    """Calculate the sigmoid of each element in the tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for the sigmoid function."""
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for sigmoid."""
        (t1,) = ctx.saved_tensors
        sigmoid_val = t1.f.mul_zip(
            t1.f.sigmoid_map(t1), (t1._ensure_tensor(1) - t1.f.sigmoid_map(t1))
        )
        return grad_output.f.mul_zip(sigmoid_val, grad_output)


class ReLU(Function):
    """ReLU function, which outputs the input directly if positive, else zero."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for ReLU."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for ReLU."""
        (t1,) = ctx.saved_tensors
        return t1.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Calculate the natural logarithm of each element in the tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for logarithm."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for logarithm."""
        (t1,) = ctx.saved_tensors
        return grad_output / t1


class Exp(Function):
    """Calculate the exponential of each element in the tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for exponentiation."""
        output = t1.f.exp_map(t1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for exponentiation."""
        (output,) = ctx.saved_tensors
        return grad_output.f.mul_zip(output, grad_output)


class Sum(Function):
    """Sum the elements of the tensor along a specified dimension."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Perform the forward pass for summation."""
        ctx.save_for_backward(t1.shape, int(dim.item()))
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Perform the backward pass for summation."""
        (t1_shape, dim) = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    """Check if elements of one tensor are less than those of another."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for less than comparison."""
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for less than, returning zeros since it's non-differentiable."""
        return grad_output._ensure_tensor(0.0), grad_output._ensure_tensor(0.0)


class EQ(Function):
    """Check if elements of two tensors are equal."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for equality check."""
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for equality check, returning zeros since it's non-differentiable."""
        return grad_output._ensure_tensor(0.0), grad_output._ensure_tensor(0.0)


class IsClose(Function):
    """Check if elements of two tensors are close to each other."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for checking closeness."""
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    """Change the order of dimensions in the tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Perform the forward pass for reordering dimensions."""
        order_list = [int(order[i]) for i in range(order.size)]
        ctx.save_for_backward(order_list)
        permuted_tensor = t1._tensor.permute(*order_list)
        new_tensor = minitorch.Tensor.make(
            permuted_tensor._storage,
            shape=tuple([t1.shape[i] for i in order_list]),
            backend=t1.backend,
        )
        return new_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Perform the backward pass for permuting dimensions."""
        (order_list,) = ctx.saved_values
        # Calculate the inverse order for permutation
        inverse_order = [0] * len(order_list)
        for i, p in enumerate(order_list):
            inverse_order[p] = i
        return grad_output.permute(*inverse_order), 0.0


class View(Function):
    """Change the shape of a tensor without changing its data."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Return a new tensor with a different shape."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Perform the backward pass for view, returning the gradient reshaped."""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    """Copy a tensor (identity operation)."""

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Return a copy of the input tensor."""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient as is."""
        return grad_output


class MatMul(Function):
    """Perform matrix multiplication between two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for matrix multiplication."""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for matrix multiplication."""
        t1, t2 = ctx.saved_tensors

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Create a tensor filled with zeros of the specified shape."""
    shape_list = [float(x) for x in shape]
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape_list)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with random values of the specified shape."""
    shape_list = [float(x) for x in shape]
    vals = [random.random() for _ in range(int(operators.prod(shape_list)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with specific data and shape.

    Args:
    ----
        ls: Data for the tensor.
        shape: Shape of the tensor.
        backend: Backend to use for the tensor.
        requires_grad: Whether to enable gradient tracking for this tensor.

    Returns:
    -------
        A new tensor with the specified data and shape.

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Create a tensor from the provided data and automatically determine its shape.

    Args:
    ----
        ls: Data for the tensor.
        backend: Backend to use for the tensor.
        requires_grad: Whether to enable gradient tracking for this tensor.

    Returns:
    -------
        A new tensor with the specified data.

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors
def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Approximate the derivative of a function at a specific argument using central difference.

    Args:
    ----
        f : A function that takes tensors as arguments and returns a single value.
        *vals : Tensors representing the function's inputs.
        arg : The index of the input tensor to differentiate.
        epsilon : A small value for calculating the difference.
        ind: The specific index in the tensor to evaluate the derivative.

    Returns:
    -------
        An approximation of the derivative of the function at the specified input tensor.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Verify if the gradients calculated using autodifferentiation match those from central difference.

    Args:
    ----
        f : The function whose gradients are being checked.
        *vals : Input tensors to the function.

    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )

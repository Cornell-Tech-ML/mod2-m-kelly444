from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    IndexingError,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Run a function on each element of a tensor."""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Make a function that applies a given operation to each element of a tensor."""
        ...

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Make a function that combines elements from two tensors using a specified operation."""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Make a function that reduces the tensor along a certain dimension using a specified operation."""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Multiply two tensors together (not implemented)."""
        raise NotImplementedError("Matrix multiplication not implemented.")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Set up the tensor backend with the provided operations.

        Args:
        ----
            ops : A class that defines operations for tensors (like map, zip, and reduce).

        """
        # Functions to apply to tensor elements
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Functions to combine two tensors
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Functions to reduce a tensor's dimensions
        self.add_reduce = ops.reduce(operators.add, start=0.0)
        self.mul_reduce = ops.reduce(operators.mul, start=1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Create a function that applies `fn` to each element of a tensor.

        Args:
        ----
            fn: A function that takes a number and returns a number.

        Returns:
        -------
            A function that applies `fn` over a tensor and can write results to an output tensor.

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)  # Create a new tensor if none provided
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Create a function that applies `fn` to matching elements of two tensors.

        Args:
        ----
            fn: A function that takes two numbers and returns a number.

        Returns:
        -------
            A function that combines elements from two tensors using `fn`.

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            c_shape = (
                shape_broadcast(a.shape, b.shape) if a.shape != b.shape else a.shape
            )
            out = a.zeros(c_shape)  # Create output tensor with the correct shape
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Create a function that reduces a tensor's values along a specific dimension.

        Args:
        ----
            fn: A function that combines two numbers and returns one number.
            start: The initial value for the reduction.

        Returns:
        -------
            A function that reduces a tensor along the specified dimension.

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1  # Set the size of the specified dimension to 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = (
                start  # Initialize the output with the starting value
            )
            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Placeholder for matrix multiplication."""
        raise NotImplementedError("Matrix multiplication not implemented.")

    is_cuda = False


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level function to apply a function to each element of a tensor.

    Args:
    ----
        fn: A function that takes a number and returns a number.

    Returns:
    -------
        A function that applies `fn` to tensor data.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        in_index = np.zeros(len(in_shape), dtype=np.int32)

        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            try:
                broadcast_index(out_index, out_shape, in_shape, in_index)
            except IndexingError:
                return

            in_pos = index_to_position(in_index, in_strides)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level function to combine elements from two tensors.

    Args:
    ----
        fn: A function that takes two numbers and returns a number.

    Returns:
    -------
        A function that combines tensor data.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        b_index = np.zeros(len(b_shape), dtype=np.int32)

        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            try:
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
            except IndexingError:
                return

            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level function to combine values of a tensor along a specific dimension.

    Args:
    ----
        fn: A function that combines two numbers into one.

    Returns:
    -------
        A function that reduces tensor data.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)

        for i in range(len(out)):
            out_shape[reduce_dim] = 1  # Set the size of the reduce dimension to 1
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            a_index = out_index.copy()

            for j in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index, a_strides)
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])  # Combine values

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)

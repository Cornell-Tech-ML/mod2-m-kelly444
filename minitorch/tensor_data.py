from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """An error that happens when there's a problem with indexing."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Turns a multi-dimensional index into a single position in a flat storage.

    Args:
    ----
        index : A tuple of integers representing the multi-dimensional index.
        strides : A list of integers that shows how to navigate the storage.

    Returns:
    -------
        The position in the flat storage corresponding to the index.

    """
    return np.dot(index, strides)


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Changes a flat position number into a multi-dimensional index.

    Args:
    ----
        ordinal: The flat position number to convert.
        shape : The shape of the tensor.
        out_index : The output where the resulting index will be stored.

    """
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = ordinal % shape[i]
        ordinal //= shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Adjusts a large index to fit into a smaller shape, following specific rules.

    Args:
    ----
        big_index : The multi-dimensional index of a larger tensor.
        big_shape : The shape of the larger tensor.
        shape : The shape of the smaller tensor.
        out_index : The output index that will fit the smaller shape.

    Returns:
    -------
        None

    """
    big_index_reverse = list(reversed(big_index))
    big_shape_reverse = list(reversed(big_shape))
    shape_reversed = list(reversed(shape))
    if len(big_shape) < len(shape):
        raise IndexingError(
            f"Dimension at index big_shape must >="
            f"dimension of shape: {len(big_shape)} "
            f"vs {len(shape)}."
        )

    for i in range(0, len(shape_reversed)):
        big_ind = big_index_reverse[i]
        small_dim = shape_reversed[i]
        big_dim = big_shape_reverse[i]
        if (small_dim != big_dim and big_dim != 1 and small_dim != 1) or (
            big_dim < small_dim
        ):
            raise IndexingError(
                f"Dimension at index big_shape must match"
                f"dimension of shape: {small_dim} "
                f"vs {big_shape_reverse[i]}."
            )
        elif small_dim == 1 and big_dim != 1:
            out_index[len(shape) - i - 1] = 0
        else:
            out_index[len(shape) - i - 1] = big_ind


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Combines two shapes to make a new shape that can fit both.

    Args:
    ----
        shape1 : The first shape.
        shape2 : The second shape.

    Returns:
    -------
        The new combined shape.

    Raises:
    ------
        IndexingError : If the shapes can't be combined.

    """
    shape1_reverse = list(reversed(shape1))
    shape2_reverse = list(reversed(shape2))

    broadcasted_shape = []
    max_dims = max(len(shape1_reverse), len(shape2_reverse))
    for i in range(0, max_dims):
        dim1 = shape1_reverse[i] if i < len(shape1) else 1
        dim2 = shape2_reverse[i] if i < len(shape2) else 1
        if dim1 == dim2 or (dim1 == 1 or dim2 == 1):
            broadcasted_shape.append(max(dim1, dim2))
        else:
            raise IndexingError(
                f"Dimension at index {max_dims-i} shape1 must match"
                f"dimension of shape2: {dim1} vs {dim2}."
            )
    return tuple(reversed(broadcasted_shape))


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Gives the step sizes needed to navigate through a shape in memory."""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initializes the tensor data with its storage, shape, and strides.

        Args:
        ----
            storage: The data to be stored in the tensor.
            shape : The shape of the tensor.
            strides : The step sizes to navigate through the tensor (optional).

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(list(shape)))  # Fix applied here
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Transfers the data to the GPU for faster processing."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Checks if the tensor's layout is organized correctly.

        Returns
        -------
            True if the layout is correct, otherwise False.

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Combines two shapes into a new shape that fits both."""
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Converts a tensor index into a single position in storage.

        Args:
        ----
            index: The index to convert. It can be a single integer for 1D tensors
                   or a tuple of integers for multi-dimensional tensors.

        Returns:
        -------
            The position in storage that corresponds to the given index.

        Raises:
        ------
            IndexingError: If the index is out of bounds or invalid.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generates all possible indices for the tensor."""
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Gets a random valid index for the tensor."""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieves the value at a specific index in the tensor."""
        return self._storage[self.index(key)]

    def set(self, key: UserIndex, val: float) -> None:
        """Sets a value at a specific index in the tensor."""
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns the core data of the tensor as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Changes the order of the dimensions of the tensor.

        Args:
        ----
            *order: The new order for the dimensions.

        Returns:
        -------
            A new TensorData with the same data but in a different order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        permutation = list(order)
        new_shape = tuple([self.shape[ind] for ind in permutation])
        new_strides = tuple([self.strides[ind] for ind in permutation])
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Converts the tensor data to a string format."""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s

"""Main setup for the Tensor object used in automatic differentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import IndexingError, TensorData

# Uncomment the following imports when implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union
    import numpy.typing as npt
    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]

@dataclass
class History:
    """Keeps track of how this variable was created through operations."""
    
    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()

_tensor_count = 0

class Tensor:
    """Represents a variable that can hold multi-dimensional data (like arrays)."""

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        self.name = name if name is not None else str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set if the tensor should track gradients.

        Args:
            x (bool): True if gradients should be tracked.
        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if this Tensor tracks gradients."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert this Tensor to a NumPy array."""
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Convert a number to a tensor using the same backend."""
        if isinstance(b, (int, float)):
            return Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            return b

    def item(self) -> float:
        """Get the value of a 1-element tensor as a float."""
        assert self.size == 1
        return self._tensor._storage[0]

    def contiguous(self) -> Tensor:
        """Get a continuous tensor with the same data."""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key_tuple = (key,) if isinstance(key, int) else key
        return self._tensor.get(key_tuple)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key_tuple = (key,) if isinstance(key, int) else key
        self._tensor.set(key_tuple, val)

    # Internal methods for automatic differentiation
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from the provided data."""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Adjust the tensor size for calculations with another tensor.

        Args:
            other: The tensor to adjust (must match or be compatible with self).

        Returns:
            The adjusted version of `other`.
        """
        # Case 1: Shapes match.
        if self.shape == other.shape:
            return other

        # Case 2: Adjust `other` to fit this tensor's shape.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)

        if self.shape == true_shape:
            return buf

        # Case 3: Adjust for different shapes.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)

        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)

        assert out.size == self.size, f"Shape mismatch: {out.shape} vs {self.shape}"
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a new tensor filled with zeros.

        Args:
            shape: The shape for the zero tensor.

        Returns:
            A zero-filled tensor.
        """
        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        return zero(shape) if shape else zero(self.shape)

    def zero_grad_(self) -> None:
        """Reset the current gradient to zero."""
        self.grad = self.zeros()

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor's data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Separate the tensor from the computation history."""
        return Tensor(self._tensor, backend=self.backend)

    # Methods for gradient calculations
    def accumulate_derivative(self, x: Any) -> None:
        """Add the gradient to this variable. Only for top-level variables.

        Args:
            x: The gradient value to add.
        """
        assert self.is_leaf(), "Only top-level variables can have gradients."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Check if this variable was created directly by the user."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the Tensor is a fixed constant."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the inputs that were used to create this Tensor."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Calculate gradients for inputs based on output gradients.

        Args:
            d_output: The output gradient for backward calculation.

        Returns:
            Pairs of (input variable, gradient).
        """
        h = self.history
        assert h is not None and h.last_fn is not None and h.ctx is not None

        gradients = h.last_fn._backward(h.ctx, d_output)
        assert len(gradients) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, gradients)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Run the backward pass for this Tensor.

        Args:
            grad_output: Optional gradient output for non-scalar Tensors.
        """
        if grad_output is None:
            assert self.shape == (1,), "Provide grad_output if not a single value."
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    # Operator overloads
    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Matrix multiplication (to be used in future modules)."""
        return MatMul.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: Tensor) -> Tensor:
        """Check if two Tensors are equal."""
        return EQ.apply(self, self._ensure_tensor(b))

    def __add__(self, b: Tensor) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: Tensor) -> Tensor:
        return Add.apply(self, Neg.apply(self._ensure_tensor(b)))

    def __mul__(self, b: Tensor) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: Tensor) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: Tensor) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: Tensor) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __rmul__(self, b: Tensor) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    # Tensor operations
    def all(self, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all elements in the Tensor are true."""
        return All.apply(self) if dim is None else All.apply(self, dim)

    def sigmoid(self) -> Tensor:
        """Apply the Sigmoid function to the Tensor."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply the ReLU function to the Tensor."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Apply the logarithm to the Tensor."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply the exponential function to the Tensor."""
        return Exp.apply(self)

    def is_close(self, b: Tensor) -> Tensor:
        """Check if two Tensors have similar values."""
        return IsClose.apply(self, self._ensure_tensor(b))

    def sum(self, dim: Optional[Tensor | int] = None) -> Tensor:
        """Calculate the sum of elements in the Tensor along a dimension.

        Args:
            dim: The dimension to sum over. If None, sum all elements.

        Returns:
            A Tensor with the sum.
        """
        if dim is None:
            one_d_tensor = self.contiguous().view(self._ensure_tensor(-1))
            return Sum.apply(one_d_tensor, Tensor.make([0], (1,), backend=self.backend))
        else:
            dim = self._ensure_tensor(dim)
            if dim.item() > self.dims:
                raise IndexingError(f"Invalid dimension: max dim is {self.dims}.")
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[Tensor | int] = None) -> Tensor:
        """Calculate the average of the Tensor over a dimension."""
        if dim is None:
            return Mul.apply(self.sum(), Inv.apply(self._ensure_tensor(self.size)))
        else:
            dim = self._ensure_tensor(dim)
            sum_tensor = self.sum(dim)
            count = self._ensure_tensor(self.shape[int(dim.item())])
            return Mul.apply(sum_tensor, Inv.apply(count))

    def permute(self, *order: int, dim: Optional[Tensor] = None) -> Tensor:
        """Change the order of the tensor's dimensions."""
        return Permute.apply(self, Tensor.make(list(order), (len(order),), backend=self.backend))

    def view(self, *shape: TensorLike, dim: Optional[Tensor] = None) -> Tensor:
        """Get a new Tensor with a different shape."""
        converted_shape = [
            int(s.item()) if isinstance(s, Tensor) else int(s) for s in shape
        ]
        if len(converted_shape) == 1 and converted_shape[0] == -1:
            return View.apply(self, self._ensure_tensor(self._tensor.size))
        return View.apply(
            self,
            Tensor.make(
                list(converted_shape), (len(converted_shape),), backend=self.backend
            ),
        )

    @property
    def shape(self) -> UserShape:
        """Get the dimensions of the tensor."""
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Get the total number of elements in the tensor."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Get how many dimensions the tensor has."""
        return self._tensor.dims

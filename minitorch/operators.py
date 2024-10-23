"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, List, TypeVar, Union

# Import Scalar from its module
from minitorch.scalar import (
    Scalar,
)  # Replace 'your_module' with the actual module where Scalar is defined

# Type variables for higher-order functions
T = TypeVar("T")
U = TypeVar("U")

# ## Task 0.1


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers together."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: Union[float, Scalar], tol: float = 1e-2) -> bool:
    """Checks if two numbers are close in value."""
    if isinstance(y, Scalar):  # Check if y is an instance of Scalar
        y_value = y.data
    else:
        y_value = y
    return abs(x - y_value) < tol


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    if x <= 0:
        raise ValueError("Logarithm of non-positive number is not defined.")
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of a number."""
    if x == 0:
        raise ZeroDivisionError("division by zero")
    return 1.0 / x


def div(x: float, y: float) -> float:
    """Divides one number by another."""
    if y == 0:
        raise ZeroDivisionError("division by zero")
    return x / y


def log_back(x: float, grad: float) -> float:
    """Computes the derivative of the logarithm function, multiplied by a second argument."""
    return grad / x


def inv_back(x: float, grad: float) -> float:
    """Computes the derivative of the reciprocal function, multiplied by a second argument."""
    return -grad / (x**2)


def relu_back(x: float, grad: float) -> float:
    """Computes the derivative of the ReLU function, multiplied by a second argument."""
    return grad if x > 0 else 0


# ## Task 0.3


def map(func: Callable[[T], U], iterable: Iterable[T]) -> List[U]:
    """Applies a function to each item in a collection."""
    return [func(x) for x in iterable]


def zipWith(
    func: Callable[[T, T], U], iterable1: Iterable[T], iterable2: Iterable[T]
) -> List[U]:
    """Combines elements from two lists using a function."""
    return [func(x, y) for x, y in zip(iterable1, iterable2)]


def reduce(func: Callable[[T, T], T], iterable: Iterable[T], initial: T) -> T:
    """Reduces a list to a single value using a function."""
    result = initial
    for item in iterable:
        result = func(result, item)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negates all numbers in a list."""
    return map(neg, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Adds corresponding numbers from two lists."""
    return zipWith(add, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Calculates the total sum of numbers in a list."""
    return reduce(add, lst, 0.0)


def prod(lst: List[float]) -> float:
    """Calculates the product of all numbers in a list."""
    return reduce(mul, lst, 1.0)

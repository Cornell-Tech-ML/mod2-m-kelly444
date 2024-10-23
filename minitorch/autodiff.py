from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Tuple, Protocol

# ## Task 1.1
# Central Difference Calculation


def central_difference(
    f: Callable[..., Any], *vals: float, arg: int = 0, epsilon: float = 1e-6
) -> float:
    """Estimate how much a function changes when you change one of its inputs slightly.

    Args:
    ----
        f: A function that takes some numbers and gives one output.
        *vals: The numbers you want to plug into the function.
        arg: Which number to slightly change (by default, the first one).
        epsilon: How much to change that number (a small value).

    Returns:
    -------
        The estimated change in the function's output based on the change in the specified input.

    Raises:
    ------
        IndexError: If you try to change a number that doesn’t exist in the list of inputs.

    """
    if arg < 0 or arg >= len(vals):
        raise IndexError(
            f"Argument index {arg} is out of bounds for {len(vals)} values."
        )

    # Create new values by slightly changing the specified input
    positive_vals = list(vals)
    positive_vals[arg] += epsilon

    negative_vals = list(vals)
    negative_vals[arg] -= epsilon

    derivative = (f(*positive_vals) - f(*negative_vals)) / (2 * epsilon)

    print(
        "Central difference result:",
        derivative,
        "Positive vals:",
        positive_vals,
        "Negative vals:",
        negative_vals,
        "Original vals:",
        vals,
        "Epsilon:",
        epsilon,
        "Function:",
        f,
    )
    return derivative


class Variable(Protocol):
    """Defines what a variable looks like in our calculation setup."""

    def accumulate_derivative(self, x: float) -> None:
        """Add a value to the total change for this variable.

        Args:
        ----
            x: The change to add.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Get a special number that identifies this variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if this variable was created by the user (not by another function)."""
        ...

    def is_constant(self) -> bool:
        """Check if this variable doesn't change (is a constant)."""
        ...

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the variables that this one depends on."""
        ...

    def chain_rule(self, d_output: float) -> Iterable[Tuple[Variable, float]]:
        """Calculate how changes in this variable affect its parents."""
        ...


def visit(variable: Variable, visited: set[int], topo_list: List[Variable]) -> None:
    """Explore a variable and its dependencies to create a list in a specific order.

    Args:
    ----
        variable: The current variable we are looking at.
        visited: A set of variables we've already checked.
        topo_list: A list that will hold the order of variables.

    """
    if variable.unique_id in visited:
        return

    for parent in variable.parents:
        if parent.unique_id not in visited and not parent.is_constant():
            visit(parent, visited, topo_list)

    visited.add(variable.unique_id)
    topo_list.append(variable)


def topological_sort(variable: Variable) -> List[Variable]:
    """Create an ordered list of variables based on their dependencies.

    Args:
    ----
        variable: The output variable from which we start.

    Returns:
    -------
        A list of variables ordered by their dependencies, from output to inputs.

    """
    visited = set()
    topo_list = []
    visit(variable, visited, topo_list)
    return topo_list


def backpropagate(variable: Variable, deriv: float) -> None:
    """Calculate how changes in the output variable affect all input variables.

    Args:
    ----
        variable: The output variable we start from.
        deriv: The initial change to propagate backward.

    """
    topo_list = topological_sort(variable)
    derivatives = {topo_list[-1].unique_id: deriv}

    # If the output variable is a leaf, just add a zero change
    if topo_list[-1].is_leaf():
        variable.accumulate_derivative(0)
        return

    # Go through the variables in reverse order and update changes
    for var in reversed(topo_list):
        if var.is_leaf():
            var.accumulate_derivative(derivatives.get(var.unique_id, 0.0))
            continue

        chain_rule_outputs = var.chain_rule(derivatives[var.unique_id])
        for parent, output in chain_rule_outputs:
            derivatives[parent.unique_id] = (
                derivatives.get(parent.unique_id, 0.0) + output
            )


@dataclass
class Context:
    """Stores information while calculating and preparing for backtracking."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Keep values that might be needed later for calculations, unless no_grad is True."""
        if not self.no_grad:
            self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the stored values from the calculations for later use."""
        return self.saved_values

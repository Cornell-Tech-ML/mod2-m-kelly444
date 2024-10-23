from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """A Module is a building block for creating neural networks."""

    def __init__(self) -> None:
        """Initializes a new Module with empty child modules and parameters, set to training mode."""
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}
        self.training: bool = True

    def modules(self) -> Sequence[Module]:
        """Retrieve the immediate child modules of this module."""
        return list(self._modules.values())

    def train(self) -> None:
        """Set this module and all its child modules to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        """Set this module and all its child modules to evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Gather all parameters from this module and its child modules."""
        params = [(name, param) for name, param in self._parameters.items()]
        for name, module in self._modules.items():
            params.extend(
                (f"{name}.{child_name}", child_param)
                for child_name, child_param in module.named_parameters()
            )
        return params

    def parameters(self) -> Sequence[Parameter]:
        """Collect all parameters from this module and its child modules."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a new parameter to this module."""
        val = Parameter(v, k)
        self._parameters[k] = val
        return val

    def __setattr__(self, key: str, val: Any) -> None:
        """Custom setter for handling parameters and child modules."""
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Custom getter for retrieving parameters and child modules."""
        if key in self._parameters:
            return self._parameters[key]
        if key in self._modules:
            return self._modules[key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Makes the module callable, running the `forward` method."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """String representation of the module and its child modules."""

        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            return first + "\n" + "\n".join(s2)

        child_lines = [
            f"({key}): " + _addindent(repr(module), 2)
            for key, module in self._modules.items()
        ]
        main_str = (
            f"{self.__class__.__name__}(\n  " + "\n  ".join(child_lines) + "\n)"
            if child_lines
            else f"{self.__class__.__name__}()"
        )
        return main_str


class Parameter:
    """A Parameter holds a value used in a Module."""

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Initialize a new Parameter with a value and an optional name."""
        self.value = x  # Store the value in a separate attribute
        self.data = x  # Retain the original data attribute for backward compatibility
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.data.requires_grad_(True)
            if self.name:
                self.data.name = self.name

    def update(self, x: Any) -> None:
        """Update the value of the parameter."""
        self.value = x  # Update the value attribute
        self.data = x  # Update the data attribute
        if hasattr(x, "requires_grad_"):
            self.data.requires_grad_(True)
            if self.name:
                self.data.name = self.name

    def __repr__(self) -> str:
        """Returns a string representation of the parameter's value."""
        return repr(self.value)  # Return the value instead of data

    def __str__(self) -> str:
        """Returns a human-readable string representation of the parameter's value."""
        return str(self.value)  # Return the value instead of data

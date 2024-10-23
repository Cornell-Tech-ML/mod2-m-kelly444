import networkx as nx
from dataclasses import dataclass
import minitorch
from typing import Any, Dict, Union

# Check if minitorch has the Scalar class
if hasattr(minitorch, "Scalar"):
    Scalar = minitorch.Scalar  # type: ignore
else:

    @dataclass
    class Scalar:
        name: str


def build_expression(code: str) -> Scalar:
    out: Scalar = eval(
        code,
        {
            "x": minitorch.Scalar(1.0, name="x"),
            "y": minitorch.Scalar(1.0, name="y"),
            "z": minitorch.Scalar(1.0, name="z"),
        },
    )
    out.name = "out"
    return out


def build_tensor_expression(code: str) -> minitorch.Tensor:
    variables: Dict[str, minitorch.Tensor] = {
        "x": minitorch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
        "y": minitorch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
        "z": minitorch.tensor([[1.0, 2.0, 3.0]], requires_grad=True),
    }
    variables["x"].name = "x"
    variables["y"].name = "y"
    variables["z"].name = "z"

    out: minitorch.Tensor = eval(code, variables)
    out.name = "out"
    return out


class GraphBuilder:
    def __init__(self) -> None:
        self.op_id: int = 0
        self.hid: int = 0
        self.intermediates: Dict[str, int] = {}

    def get_name(self, x: Union[Scalar, minitorch.Tensor]) -> str:
        if not isinstance(x, Scalar) and not isinstance(x, minitorch.Tensor):
            return "constant %s" % (x,)
        elif len(x.name) > 15:
            if x.name in self.intermediates:
                return "v%d" % (self.intermediates[x.name],)
            else:
                self.hid += 1
                self.intermediates[x.name] = self.hid
                return "v%d" % (self.hid,)
        else:
            return x.name

    def run(self, final: Union[Scalar, minitorch.Tensor]) -> nx.MultiDiGraph:
        queue: list[list[Union[Scalar, minitorch.Tensor]]] = [[final]]
        G: nx.MultiDiGraph = nx.MultiDiGraph()
        G.add_node(self.get_name(final))

        while queue:
            (cur,) = queue[0]
            queue = queue[1:]

            # Check if cur is a minitorch.Tensor
            if isinstance(cur, minitorch.Tensor):
                if cur.is_constant() or cur.is_leaf():
                    continue

                # Ensure cur has the history attribute
                if not hasattr(cur, "history") or cur.history is None:
                    continue

                op_name = (
                    cur.history.last_fn.__name__
                    if cur.history.last_fn is not None
                    else "unknown_op"
                )
                op: str = f"{op_name} (Op {self.op_id})"
                G.add_node(op, shape="square", penwidth=3)
                G.add_edge(op, self.get_name(cur))
                self.op_id += 1

                # Ensure inputs is a valid attribute of history
                if hasattr(cur.history, "inputs"):
                    for i, input in enumerate(cur.history.inputs):
                        G.add_edge(self.get_name(input), op, f"{i}")
                        queue.append([input])  # Append inputs for further processing

        return G

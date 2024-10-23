import networkx as nx
import minitorch
from typing import Union, Any
from networkx.drawing.nx_pydot import to_pydot


def expression() -> minitorch.Scalar:
    """Create an autodiff expression."""
    x = minitorch.Scalar(1.0, name="x")
    y = minitorch.Scalar(1.0, name="y")
    z = (x * x) * y + 10.0 * x
    z.name = "z"
    return z


class GraphBuilder:
    """Build a graph representation of the computation."""

    def __init__(self) -> None:
        self.op_id = 0
        self.hid = 0
        self.intermediates: dict[str, int] = {}

    def get_name(self, x: Union[minitorch.Scalar, float]) -> str:
        """Get a display name for a value in the computation graph."""
        if not isinstance(x, minitorch.Scalar):
            return f"constant {x}"
        elif len(x.name) > 15:
            if x.name in self.intermediates:
                return f"v{self.intermediates[x.name]}"
            else:
                self.hid += 1
                self.intermediates[x.name] = self.hid
                return f"v{self.hid}"
        else:
            return x.name

    def run(self, final: minitorch.Scalar) -> nx.MultiDiGraph:
        """Build a computational graph from a final scalar value."""
        queue = [[final]]
        G = nx.MultiDiGraph()
        G.add_node(self.get_name(final))

        while queue:
            (cur,) = queue[0]
            queue = queue[1:]

            if cur.history is None or cur.is_leaf():
                continue

            # Create operation node
            if cur.history.last_fn is not None:
                op = f"{cur.history.last_fn.__name__} (Op {self.op_id})"
            else:
                op = f"Unknown Operation (Op {self.op_id})"

            G.add_node(op, shape="square", penwidth=3)
            G.add_edge(op, self.get_name(cur))
            self.op_id += 1

            # Add edges for operation inputs
            for i, input_val in enumerate(cur.history.inputs):
                G.add_edge(self.get_name(input_val), op, f"{i}")

            # Queue up input scalars for processing
            for input_val in cur.history.inputs:
                if not isinstance(input_val, minitorch.Scalar):
                    continue
                if not any(s[0] == input_val for s in queue):
                    queue.append([input_val])

        return G


def make_graph(y: minitorch.Scalar, lr: bool = False) -> bytes:
    """
    Create a graphviz visualization of the computation graph.

    Args:
        y: The final scalar value in the computation
        lr: If True, render graph from left to right instead of top to bottom

    Returns:
        SVG visualization of the graph as bytes
    """
    G = GraphBuilder().run(y)
    if lr:
        G.graph["graph"] = {"rankdir": "LR"}  # type: ignore
    output_graphviz_svg = to_pydot(G).create_svg()
    return output_graphviz_svg

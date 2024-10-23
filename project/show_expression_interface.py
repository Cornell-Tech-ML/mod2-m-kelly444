from project import graph_builder
import networkx as nx
import streamlit as st
from streamlit_ace import st_ace
from typing import Optional, cast
from typing import Any, Dict, TypeVar, Union
import networkx as nx


class TypedMultiDiGraph(nx.MultiDiGraph):
    graph: Dict[str, Any]


def set_graph_attr(
    G: Union[nx.MultiDiGraph, TypedMultiDiGraph], attr: Dict[str, Any]
) -> TypedMultiDiGraph:
    """Helper function to set graph attributes with proper typing"""
    # Cast to TypedMultiDiGraph at the start
    typed_G = cast(TypedMultiDiGraph, G)
    if not hasattr(typed_G, "graph"):
        typed_G.graph = {}
    typed_G.graph.update(attr)
    return typed_G


def render_show_expression(tensor: bool = False) -> None:
    """Render an expression visualization interface using Streamlit.

    Args:
        tensor: If True, build tensor expressions. If False, build regular expressions.
    """
    # Initialize with default code
    if tensor:
        default_code = "(x * x) * y + 10.0 * x.sum()"
        st.text("Build an expression of tensors x, y, and z. (All the same shape)")
    else:
        default_code = "(x * x) * y + 10.0 * x"

    # Build and display initial expression
    initial_out = (
        graph_builder.build_tensor_expression(default_code)
        if tensor
        else graph_builder.build_expression(default_code)
    )
    initial_G = cast(TypedMultiDiGraph, graph_builder.GraphBuilder().run(initial_out))
    set_graph_attr(initial_G, {"rankdir": "LR"})
    st.graphviz_chart(nx.nx_pydot.to_pydot(initial_G).to_string())

    # Get user input code
    code: str = st_ace(language="python", height=300, value=default_code)

    # Build the expression graph from user input
    if tensor:
        out = graph_builder.build_tensor_expression(code)
    else:
        out = graph_builder.build_expression(code)

    # Create and configure the graph
    G = cast(TypedMultiDiGraph, graph_builder.GraphBuilder().run(out))
    set_graph_attr(G, {"rankdir": "LR"})

    # Display the updated graph
    st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

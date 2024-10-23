import networkx as nx
import streamlit as st
from streamlit_ace import st_ace
import minitorch

MyModule = None
from typing import Any, Dict, TypeVar, Union
import networkx as nx


class TypedMultiDiGraph(nx.MultiDiGraph):
    graph: Dict[str, Any]


def set_graph_attr(
    G: Union[nx.MultiDiGraph, TypedMultiDiGraph], attr: Dict[str, Any]
) -> TypedMultiDiGraph:
    """Helper function to set graph attributes with proper typing"""
    if not hasattr(G, "graph"):
        G.graph = {}
    G.graph.update(attr)
    return G  # type: ignore


def render_module_sandbox():
    G = TypedMultiDiGraph()
    set_graph_attr(G, {"rankdir": "TB"})
    st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())
    st.write("## Sandbox for Module Trees")

    st.write(
        "Visual debugging checks showing the module tree that your code constructs."
    )

    code = st_ace(
        language="python",
        height=300,
        value="""
class MyModule(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.parameter1 = minitorch.Parameter(15)
""",
    )

    # Execute the code and capture the class
    exec(code, globals())

    # Ensure MyModule is defined before using it
    if MyModule is not None:
        out = MyModule()
        st.write(dict(out.named_parameters()))

        G = nx.MultiDiGraph()
        G.add_node("base")
        stack = [(out, "base")]

        while stack:
            n, name = stack[0]
            stack = stack[1:]

            # Safely access _parameters and _modules
            parameters = getattr(n, "_parameters", {})
            modules = getattr(n, "_modules", {})

            for pname, p in parameters.items():
                G.add_node(name + "." + pname, shape="rect", penwidth=0.5)
                G.add_edge(name, name + "." + pname)

            for cname, m in modules.items():
                G.add_edge(name, name + "." + cname)
                stack.append((m, name + "." + cname))

        G.graph["graph"] = {"rankdir": "TB"}
        st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())
    else:
        st.error("MyModule is not defined. Please check your code.")

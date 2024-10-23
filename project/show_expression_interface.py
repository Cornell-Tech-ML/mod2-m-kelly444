from project import graph_builder
import networkx as nx
import streamlit as st
from streamlit_ace import st_ace
from typing import Optional


def render_show_expression(tensor: bool = False) -> None:
    if tensor:
        st.text("Build an expression of tensors x, y, and z. (All the same shape)")
        code: str = st_ace(
            language="python", height=300, value="(x * x) * y + 10.0 * x.sum()"
        )
        out = graph_builder.build_tensor_expression(code)
    else:
        code = st_ace(language="python", height=300, value="(x * x) * y + 10.0 * x")
        out = graph_builder.build_expression(code)

    G: nx.DiGraph = graph_builder.GraphBuilder().run(out)
    G.graph["graph"] = {
        "rankdir": "LR"
    }  # error: Cannot access attribute "graph" for class "MultiDiGraph[Unknown]"
    st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

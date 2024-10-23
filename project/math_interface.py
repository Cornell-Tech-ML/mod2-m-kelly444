from project import graph_builder
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from project.interface.streamlit_utils import render_function
from typing import cast

import minitorch
from minitorch.tensor import Tensor
from minitorch import MathTest, MathTestVariable
from typing import List, Tuple, Union, Any, Callable

MyModule = None
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


def render_math_sandbox(use_scalar: bool = False, use_tensor: bool = False) -> None:
    out = None  # Initialize out at the start

    if out is not None:
        G = cast(TypedMultiDiGraph, graph_builder.GraphBuilder().run(out))
        set_graph_attr(G, {"rankdir": "LR"})
        st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

    st.write("## Sandbox for Math Functions")
    st.write("Visualization of the mathematical tests run on the underlying code.")

    if use_scalar:
        one, two, red = MathTestVariable._comp_testing()
    else:
        one, two, red = MathTest._comp_testing()

    f_type: str = st.selectbox("Function Type", ["One Arg", "Two Arg", "Reduce"])
    select: dict[str, List[Tuple[str, Union[Callable, Any], Any]]] = {
        "One Arg": one,
        "Two Arg": two,
        "Reduce": red,
    }

    fn: Tuple[str, Union[Callable, Any], Any] = st.selectbox(
        "Function", select[f_type], format_func=lambda a: a[0]
    )
    name, _, scalar = fn

    if f_type == "One Arg":
        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x)")

        xs: List[float] = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]

        if use_scalar:
            if use_tensor:
                ys: List[float] = [
                    scalar(Tensor.make([p], (1,), backend=minitorch.Tensor.backend))[
                        0
                    ].item()
                    for p in xs
                ]
            else:
                ys = [scalar(minitorch.Scalar(p)).data for p in xs]
        else:
            ys = [scalar(p) for p in xs]

        scatter = go.Scatter(mode="lines", x=xs, y=ys)
        fig = go.Figure(scatter)
        st.write(fig)

        if use_scalar:
            st.write("Derivative f'(x)")
            if use_tensor:
                x_var: List[Tensor] = [
                    Tensor.make([x], (1,), backend=minitorch.Tensor.backend) for x in xs
                ]
            else:
                x_var: List[Tensor] = [
                    Tensor.make([float(x)], (1,), backend=minitorch.Tensor.backend)
                    for x in xs
                ]

            out = None  # Initialize out
            for x in x_var:
                out = scalar(x)
                if out is not None:
                    if use_tensor:
                        out.backward(
                            Tensor.make([1.0], (1,), backend=minitorch.Tensor.backend)
                        )
                    else:
                        out.backward()

            if use_tensor:
                scatter = go.Scatter(
                    mode="lines",
                    x=xs,
                    y=[x.grad.item() if x.grad else 0.0 for x in x_var],
                )
            else:
                scatter = go.Scatter(
                    mode="lines", x=xs, y=[getattr(x, "derivative", 0.0) for x in x_var]
                )

            fig = go.Figure(scatter)
            st.write(fig)

            if out is not None:
                G = cast(TypedMultiDiGraph, graph_builder.GraphBuilder().run(out))
                set_graph_attr(G, {"rankdir": "LR"})
                st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

    if f_type == "Two Arg":
        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x, y)")

        xs: List[float] = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys: List[float] = [((y / 1.0) - 50.0 + 1e-5) for y in range(1, 100)]

        if use_scalar:
            if use_tensor:
                zs: List[List[float]] = [
                    [
                        scalar(
                            Tensor.make([x], (1,), backend=minitorch.Tensor.backend),
                            Tensor.make([y], (1,), backend=minitorch.Tensor.backend),
                        )[0].item()
                        for x in xs
                    ]
                    for y in ys
                ]
            else:
                zs = [
                    [scalar(minitorch.Scalar(x), minitorch.Scalar(y)).data for x in xs]
                    for y in ys
                ]
        else:
            zs = [[scalar(x, y) for x in xs] for y in ys]

        scatter = go.Surface(x=xs, y=ys, z=zs)
        fig = go.Figure(scatter)
        st.write(fig)

        if use_scalar:
            a: List[List[Tuple[float, float, float]]] = []
            b: List[List[Tuple[float, float, float]]] = []

            for x in xs:
                oa: List[Tuple[float, float, float]] = []
                ob: List[Tuple[float, float, float]] = []

                if use_tensor:
                    for y in ys:
                        x1 = Tensor.make(
                            [x],
                            (1,),
                            backend=minitorch.Tensor.backend,
                        )
                        y1 = Tensor.make(
                            [y],
                            (1,),
                            backend=minitorch.Tensor.backend,
                        )
                        out = scalar(x1, y1)
                        if out is not None:
                            out.backward(
                                Tensor.make(
                                    [1.0], (1,), backend=minitorch.Tensor.backend
                                )
                            )
                            oa.append((x, y, x1.grad.item() if x1.grad else 0.0))
                            ob.append((x, y, y1.grad.item() if y1.grad else 0.0))
                else:
                    for y in ys:
                        x1 = minitorch.Scalar(x)
                        y1 = minitorch.Scalar(y)
                        out = scalar(x1, y1)
                        if out is not None:
                            out.backward()
                            oa.append((x, y, getattr(x1, "derivative", 0.0)))
                            ob.append((x, y, getattr(y1, "derivative", 0.0)))

                a.append(oa)
                b.append(ob)

            st.write("Derivative f'_x(x, y)")
            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in a],
                y=[[c[1] for c in a2] for a2 in a],
                z=[[c[2] for c in a2] for a2 in a],
            )
            fig = go.Figure(scatter)
            st.write(fig)

            st.write("Derivative f'_y(x, y)")
            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in b],
                y=[[c[1] for c in a2] for a2 in b],
                z=[[c[2] for c in a2] for a2 in b],
            )
            fig = go.Figure(scatter)
            st.write(fig)

    if f_type == "Reduce":
        st.write("### " + name)
        render_function(scalar)

        xs: List[float] = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys: List[float] = [((y / 1.0) - 50.0 + 1e-5) for y in range(1, 100)]

        if use_tensor:
            scatter = go.Surface(
                x=xs,
                y=ys,
                z=[
                    [
                        scalar(
                            Tensor.make([x, y], (2,), backend=minitorch.Tensor.backend)
                        )[0].item()
                        for x in xs
                    ]
                    for y in ys
                ],
            )
        else:
            scatter = go.Surface(x=xs, y=ys, z=[[scalar(x, y) for x in xs] for y in ys])
        fig = go.Figure(scatter)
        st.write(fig)

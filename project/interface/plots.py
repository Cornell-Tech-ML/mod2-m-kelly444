import plotly.graph_objects as go
from typing import List, Optional, Callable, Any


def make_scatters(
    graph: Any,
    model: Optional[Callable[[List[List[float]]], List[float]]] = None,
    size: int = 50,
) -> List[go.Figure]:
    """
    Creates scatter and contour plots based on the provided graph and model.

    Parameters:
    - graph: The graph data containing coordinates and labels.
    - model: An optional callable that takes in coordinates and returns values for contour plotting.
    - size: The resolution for the contour grid.

    Returns:
    - A list of Plotly Figures containing the generated scatter and contour plots.
    """
    color_map = ["#69bac9", "#ea8484"]
    symbol_map = ["circle-dot", "x"]
    colors = [color_map[y] for y in graph.y]
    symbols = [symbol_map[y] for y in graph.y]
    scatters = []

    if model is not None:
        colorscale = [[0, "#69bac9"], [1.0, "#ea8484"]]
        z = [
            model([[j / (size + 1.0), k / (size + 1.0)] for j in range(size + 1)])
            for k in range(size + 1)
        ]
        scatters.append(
            go.Contour(
                z=z,
                dx=1 / size,
                x0=0,
                dy=1 / size,
                y0=0,
                zmin=0.2,
                zmax=0.8,
                line_smoothing=0.5,
                colorscale=colorscale,
                opacity=0.6,
                showscale=False,
            )
        )
    scatters.append(
        go.Scatter(
            mode="markers",
            x=[p[0] for p in graph.X],
            y=[p[1] for p in graph.X],
            marker_symbol=symbols,
            marker_color=colors,
            marker=dict(size=15, line=dict(width=3, color="Black")),
        )
    )
    return scatters


def animate(self: Any, models: List[Optional[Callable]], names: List[str]) -> None:
    """
    Creates an animated plot based on the provided models and their names.

    Parameters:
    - self: The graph object to animate.
    - models: A list of optional callable models.
    - names: A list of names corresponding to each model for labeling.
    """
    scatters = [make_scatters(self, m) for m in models]
    background = [s[0] for s in scatters]
    points = scatters[0][1]

    # Create a list to hold visibility states
    visibility = [False] * (len(background) + 1)  # +1 for points
    visibility[0] = True  # Show the first background by default

    steps = []
    for i in range(len(background)):
        step = dict(
            method="update",
            args=[
                {"visible": visibility[:i] + [True] + visibility[i + 1 :]},
                {},
            ],
            label="%1.3f" % names[i],
        )
        steps.append(step)

    sliders = [
        dict(active=0, currentvalue={"prefix": "b="}, pad={"t": 50}, steps=steps)
    ]

    fig = go.Figure(
        data=background + [points],
    )
    fig.update_layout(sliders=sliders)

    fig.update_layout(
        template="simple_white",
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
    )
    fig.show()


def make_oned(
    graph: Any,
    model: Optional[Callable[[List[List[float]]], List[float]]] = None,
    size: int = 50,
) -> List[go.Figure]:
    """
    Creates a 1D scatter plot based on the provided graph and optional model.

    Parameters:
    - graph: The graph data containing coordinates and labels.
    - model: An optional callable that takes in coordinates and returns values for line plotting.
    - size: The resolution for the line plot.

    Returns:
    - A list of Plotly Figures containing the generated 1D scatter plot.
    """
    scatters = []
    color_map = ["#69bac9", "#ea8484"]
    symbol_map = ["circle-dot", "x"]
    colors = [color_map[y] for y in graph.y]
    symbols = [symbol_map[y] for y in graph.y]

    if model is not None:
        y = model([[j / (size + 1.0), 0.0] for j in range(size + 1)])
        scatters.append(
            go.Scatter(
                mode="lines",
                x=[j / (size + 1.0) for j in range(size + 1)],
                y=y,
                marker=dict(size=15, line=dict(width=3, color="Black")),
            )
        )
    scatters.append(
        go.Scatter(
            mode="markers",
            x=[p[0] for p in graph.X],
            y=graph.y,
            marker_symbol=symbols,
            marker_color=colors,
            marker=dict(size=15, line=dict(width=3, color="Black")),
        )
    )
    return scatters


def plot_out(
    graph: Any,
    model: Optional[Callable[[List[List[float]]], List[float]]] = None,
    name: str = "",
    size: int = 50,
    oned: bool = False,
) -> go.Figure:
    """
    Creates a figure based on 2D or 1D data.

    Parameters:
    - graph: The graph data to plot.
    - model: An optional callable model for plotting.
    - name: The name of the plot (not currently used).
    - size: The resolution for the plot.
    - oned: Boolean indicating whether to create a 1D plot.

    Returns:
    - A Plotly Figure containing the generated plot.
    """
    if oned:
        scatters = make_oned(graph, model, size=size)
    else:
        scatters = make_scatters(graph, model, size=size)

    fig = go.Figure(scatters)
    fig.update_layout(
        xaxis={
            "showgrid": False,
            "visible": False,
            "range": [0, 1],
        },
        yaxis={
            "showgrid": False,
            "visible": False,
            "range": [0, 1],
        },
    )
    return fig


def plot(
    graph: Any,
    model: Optional[Callable[[List[List[float]]], List[float]]] = None,
    name: str = "",
) -> None:
    """
    Displays the plot created by the plot_out function.

    Parameters:
    - graph: The graph data to plot.
    - model: An optional callable model for plotting.
    - name: The name of the plot (not currently used).
    """
    plot_out(graph, model, name).show()


def plot_function(
    title: str,
    fn: Callable[[float], float],
    arange: List[float] = [(i / 10.0) - 5 for i in range(0, 100)],
    fn2: Optional[Callable[[float], float]] = None,
) -> None:
    """
    Plots a 2D function and optionally a second function.

    Parameters:
    - title: The title of the plot.
    - fn: The primary function to plot.
    - arange: The range of x values to evaluate the function over.
    - fn2: An optional second function to plot.
    """
    ys = [fn(x) for x in arange]
    scatters = []
    scatter = go.Scatter(x=arange, y=ys)
    scatters.append(scatter)

    if fn2 is not None:
        ys = [fn2(x) for x in arange]
        scatter2 = go.Scatter(x=arange, y=ys)
        scatters.append(scatter2)

    fig = go.Figure(scatters)
    fig.update_layout(template="simple_white", title=title)

    fig.show()


def plot_function3D(
    title: str,
    fn: Callable[[float, float], float],
    arange: List[float] = [(i / 5.0) - 4.0 for i in range(0, 40)],
) -> None:
    """
    Plots a 3D surface based on a two-variable function.

    Parameters:
    - title: The title of the plot.
    - fn: The function to evaluate.
    - arange: The range of x and y values to evaluate the function over.
    """
    xs = [((x / 10.0) - 5.0 + 1e-5) for x in range(1, 100)]
    ys = [((x / 10.0) - 5.0 + 1e-5) for x in range(1, 100)]
    zs = [[fn(x, y) for x in xs] for y in ys]

    scatter = go.Surface(x=xs, y=ys, z=zs)

    fig = go.Figure(scatter)
    fig.update_layout(template="simple_white", title=title)

    fig.show()

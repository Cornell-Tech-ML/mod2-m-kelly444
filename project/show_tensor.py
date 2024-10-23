import numpy as np
import plotly.graph_objects as go
from typing import List, Optional


def tensor_figure(
    depth: int,
    columns: int,
    rows: int,
    highlighted_position: Optional[int],  # Changed to Optional[int]
    title: str = "3D Tensor Visualization",
    axis_titles: Optional[List[str]] = None,  # Changed to Optional[List[str]]
    show_fig: bool = True,
    slider: bool = True,
) -> go.Figure:
    """Create a 3D scatter plot for a tensor."""

    if axis_titles is None:
        axis_titles = ["Depth (i)", "Columns (k)", "Rows (j)"]

    fig = go.Figure()

    # Generate coordinates for the 3D tensor
    tensor_coords = np.array(
        [[i, j, k] for k in range(depth) for j in range(columns) for i in range(rows)]
    )

    fig.add_trace(
        go.Scatter3d(
            x=tensor_coords[:, 0],
            y=tensor_coords[:, 1],
            z=tensor_coords[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue", opacity=0.8),
        )
    )

    # Highlight the selected position
    if highlighted_position is not None:
        selected_coord = tensor_coords[highlighted_position]
        fig.add_trace(
            go.Scatter3d(
                x=[selected_coord[0]],
                y=[selected_coord[1]],
                z=[selected_coord[2]],
                mode="markers",
                marker=dict(size=10, color="red", opacity=1.0),
                name="Highlighted Position",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2],
        ),
        width=800,
        height=800,
    )

    return fig

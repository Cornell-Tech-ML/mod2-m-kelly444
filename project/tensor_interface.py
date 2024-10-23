import numpy as np
import plotly.graph_objects as go
import streamlit as st
from project.interface.streamlit_utils import render_function
from project.show_tensor import tensor_figure
from minitorch import (
    Tensor,
    index_to_position,
    to_index,
    TensorBackend,
    TensorOps,  # Ensure you import the correct ops class
)
from typing import List, Tuple, Union, Any


def safe_int_conversion(value: Any) -> int:
    """Safely convert a value to an integer, handling various types."""
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"Cannot convert {value} to int.")


def get_tensor_shape_input(shape: Tuple[int, ...]) -> List[int]:
    """Get index inputs for each dimension of the tensor shape."""
    return [
        safe_int_conversion(
            st.number_input(
                f"Dimension {i} index:", min_value=0, max_value=dim - 1, value=0
            )
        )
        for i, dim in enumerate(shape)
    ]


def display_tensor_storage(
    tensor: Tensor, selected_index: int, max_size: int = 10
) -> None:
    """Visualize tensor storage with a Plotly scatter plot."""
    storage = tensor._tensor._storage
    display_data = storage[:max_size] if len(storage) > max_size else storage

    fig = go.Figure(
        data=[
            go.Scatter(
                x=list(range(len(display_data))),
                y=[0] * len(display_data),
                mode="markers+text",
                text=display_data,
                textposition="middle center",
                marker=dict(
                    size=50,
                    color=[
                        "#69BAC9" if i == selected_index else "lightgray"
                        for i in range(len(display_data))
                    ],
                ),
            )
        ]
    )

    fig.update_layout(
        title="Tensor Storage Visualization",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=125,
        margin=dict(l=25, r=25, t=0, b=0),
    )

    st.write(fig)


def visualize_tensor(tensor: Tensor, index: List[int]) -> None:
    """Visualize the tensor value at a specific index."""
    if len(tensor.shape) != 3:
        st.error("Only 3D tensors can be visualized.")
        return

    position = index_to_position(
        np.array(index, dtype=np.int32),
        np.array(tensor._tensor.strides, dtype=np.int32),
    )
    value = tensor._tensor._storage[
        safe_int_conversion(position)
    ]  # Ensure position is an int

    st.write(f"**Value at index {index}:** {value}")

    fig = tensor_figure(
        tensor.shape[0],
        tensor.shape[2],
        tensor.shape[1],
        safe_int_conversion(position),
        f"Storage Position: {position}, Index: {index}",
        show_fig=False,
    )
    st.write(fig)


def visualize_tensor_interface(tensor: Tensor) -> None:
    """Main interface to visualize tensor and its storage."""
    st.write(f"**Tensor Strides:** {tensor._tensor.strides}")
    selected_index = safe_int_conversion(
        st.slider(
            "Select Storage Position", 0, len(tensor._tensor._storage) - 1, value=0
        )
    )

    tensor_shape = tensor.shape
    out_index = get_tensor_shape_input(tuple(tensor_shape))  # Ensure tuple is passed

    visualize_tensor(tensor, out_index)
    display_tensor_storage(tensor, selected_index)


def interface_index_to_position(tensor: Tensor) -> None:
    """Interface for the index_to_position function."""
    index_input = st.text_input(
        "Multi-dimensional index", value=str([0] * len(tensor.shape))
    )
    strides_input = st.text_input(
        "Tensor Strides", value=str(list(tensor._tensor.strides))
    )

    try:
        index = eval(index_input)
        strides = eval(strides_input)

        if isinstance(index, (list, tuple)) and len(index) == len(tensor.shape):
            position = index_to_position(
                np.array(index, dtype=np.int32), np.array(strides, dtype=np.int32)
            )
            st.write(f"**Position in Storage:** {position}")
        else:
            st.error("Invalid index format.")
    except Exception as e:
        st.error(f"Error: {e}")


def interface_to_index(tensor: Tensor) -> None:
    """Interface for the to_index function."""
    position = safe_int_conversion(
        st.number_input("Storage Position", 0, len(tensor._tensor._storage) - 1)
    )
    out_index = np.zeros(len(tensor.shape), dtype=np.int32)
    to_index(position, np.array(tensor.shape, dtype=np.int32), out_index)
    st.write(f"**Index corresponding to position {position}:** {out_index.tolist()}")


def interface_permute(tensor: Tensor) -> None:
    """Interface for tensor permutation."""
    permutation_input = st.text_input(
        "Permutation", value=str(list(range(len(tensor.shape)))[::-1])
    )
    try:
        permutation = eval(permutation_input)
        permuted_tensor = tensor.permute(*permutation)
        st.write(f"**Permuted Tensor Strides:** {permuted_tensor._tensor.strides}")
    except Exception as e:
        st.error(f"Error: {e}")


def render_tensor_sandbox() -> None:
    """Sandbox for creating and visualizing tensors."""
    st.write("## Tensor Sandbox")
    tensor_shape = eval(st.text_input("Tensor Shape", value="(2, 2, 2)"))

    random_data = st.checkbox("Fill Tensor with Random Numbers", value=True)
    tensor_data = np.round(
        np.random.rand(*tensor_shape)
        if random_data
        else np.arange(np.prod(tensor_shape)),
        2,
    )

    try:
        # Create a TensorBackend instance with the necessary arguments
        backend_instance = TensorBackend(
            ops=TensorOps
        )  # Ensure you use a valid ops class

        tensor = Tensor.make(
            tensor_data.tolist(), tensor_shape, backend=backend_instance
        )  # Use the instance
        st.write("**Tensor Created Successfully!**")

        action = st.selectbox(
            "Select Action",
            ["Visualize Tensor", "Index to Position", "To Index", "Permute Tensor"],
        )

        if action == "Visualize Tensor":
            visualize_tensor_interface(tensor)
        elif action == "Index to Position":
            interface_index_to_position(tensor)
        elif action == "To Index":
            interface_to_index(tensor)
        elif action == "Permute Tensor":
            interface_permute(tensor)
    except Exception as e:
        st.error(f"Failed to create tensor: {e}")


# Call the main function to run the sandbox
render_tensor_sandbox()

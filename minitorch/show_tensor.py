# show_tensor.py
import streamlit as st
from minitorch import Tensor
from minitorch.tensor_visualization import st_visualize_tensor, st_visualize_storage

from typing import Sequence  # Use Sequence instead of List


def index_to_position(index: int, strides: Sequence[int]) -> int:
    """Convert a given index into a storage position based on tensor strides."""
    position = 0
    for i in range(len(strides)):
        position += index * strides[i]
        index = 0  # Only the first index matters for this calculation
    return position


def show_tensor(tensor: Tensor) -> None:
    """Visualize a tensor in Streamlit."""
    st.write("## Tensor Visualization")

    # Display tensor shape and strides
    st.write(f"**Tensor Shape:** {tensor.shape}")
    st.write(f"**Tensor Strides:** {tensor._tensor.strides}")

    # Allow user to select an index
    index = st.selectbox("Select index for visualization", range(len(tensor.shape)))

    # Calculate the corresponding storage position
    storage_position = index_to_position(index, tensor._tensor.strides)

    # Show the value at the selected index
    st.write(f"**Value at index {index}:** {tensor._tensor._storage[storage_position]}")

    # Visualize the tensor and its storage
    st_visualize_tensor(tensor, index)
    st_visualize_storage(tensor, storage_position)

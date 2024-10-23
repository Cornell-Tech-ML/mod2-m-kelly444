# tensor_visualization.py
import streamlit as st
from minitorch import Tensor  # Assuming you have a Tensor class to work with


def st_visualize_tensor(tensor: Tensor, index: int) -> None:
    """Visualize the tensor."""
    st.write(f"Visualizing tensor at index {index}:")
    # You can add more visualization logic here, like plotting or displaying values.


def st_visualize_storage(tensor: Tensor, storage_position: int) -> None:
    """Visualize the tensor's storage."""
    st.write(f"Visualizing storage at position {storage_position}:")
    # Add your storage visualization logic here.

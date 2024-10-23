import streamlit as st
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


# Define a simple neural network for fast training
class FastTrain(nn.Module):
    def __init__(self) -> None:
        super(FastTrain, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data() -> DataLoader:
    """Load the MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    return train_loader


def train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 5) -> None:
    """Train the neural network on the MNIST dataset efficiently."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def tensor_operations_demo(
    size: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Perform various tensor operations and measure execution time."""
    a = torch.rand(size)
    b = torch.rand(size)

    # Measure time for addition
    start_time = time.time()
    c = a + b
    add_time = time.time() - start_time

    # Measure time for multiplication
    start_time = time.time()
    d = a * b
    mul_time = time.time() - start_time

    return c, d, add_time, mul_time


def render_run_fast_tensor_interface() -> None:
    """Render the fast tensor operations and training interface."""
    st.header("Fast Tensor Operations and Model Training")

    # Load data
    train_loader = load_data()
    model = FastTrain()

    # User input for epochs using number input
    epochs = int(
        st.sidebar.number_input("Number of epochs", min_value=1, value=5, step=1)
    )

    # Button to train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            train_model(model, train_loader, epochs=epochs)
        st.success("Model trained successfully!")

    # User input for tensor size
    size = int(
        st.sidebar.number_input("Enter tensor size:", min_value=1, value=1000, step=1)
    )

    # Button to perform tensor operations
    if st.button("Run Tensor Operations"):
        with st.spinner("Running operations..."):
            results = tensor_operations_demo(size)
            c, d, add_time, mul_time = results

        # Display results
        st.subheader("Results")
        st.write(f"Addition result (first 5 elements): {c[:5]}")
        st.write(f"Multiplication result (first 5 elements): {d[:5]}")
        st.write(f"Time taken for addition: {add_time:.6f} seconds")
        st.write(f"Time taken for multiplication: {mul_time:.6f} seconds")

        # Plotting results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(c.numpy(), label="Addition Result")
        axes[0].set_title("Addition Result")
        axes[0].legend()

        axes[1].plot(d.numpy(), label="Multiplication Result", color="orange")
        axes[1].set_title("Multiplication Result")
        axes[1].legend()

        st.pyplot(fig)


# Make this function callable from the main app
if __name__ == "__main__":
    render_run_fast_tensor_interface()

import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Iterator, Tuple


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super(SimpleNN, self).__init__()
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
    """Train the neural network on the MNIST dataset."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def plot_sample_images(train_loader: DataLoader) -> None:
    """Plot sample images from the training set."""
    data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]] = iter(train_loader)
    images, labels = next(data_iter)
    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    for ax, img, label in zip(axes, images[:5], labels[:5]):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Label: {label.item()}")
    st.pyplot(fig)


def render_run_image_interface() -> None:
    """Render the MNIST image display interface."""
    st.header("Visualize MNIST Images")

    # Load data
    train_loader = load_data()

    # Button to display sample images
    if st.button("Show Sample Images"):
        plot_sample_images(train_loader)


def render_run_mnist_interface() -> None:
    """Render the MNIST training interface."""
    st.header("MNIST Handwritten Digit Classification")

    # Load data
    train_loader = load_data()

    # Initialize model
    model = SimpleNN()

    # Set training parameters
    epochs = st.sidebar.slider("Number of epochs", min_value=1, value=5, step=1)

    # Ensure epochs is an integer
    if not isinstance(epochs, int):
        st.error("Epochs value must be an integer.")
        epochs = 5  # Default to 5 if there's an error

    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training..."):
            train_model(model, train_loader, epochs=epochs)
        st.success("Model trained successfully!")

    # Visualize images option
    if st.checkbox("Visualize MNIST Images"):
        render_run_image_interface()


# Make this function callable from the main app
if __name__ == "__main__":
    render_run_mnist_interface()

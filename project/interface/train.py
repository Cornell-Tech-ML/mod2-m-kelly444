import time
from typing import Type, Any, List, Dict
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import minitorch
import pandas as pd
from datetime import datetime, date, time as dt_time

from project import graph_builder
import project.interface.plots as plots


def sanitize_value(value: Any) -> float:
    """Sanitize a single value, converting it to float."""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, datetime):
        return float(value.timestamp())
    elif isinstance(value, (tuple, list)):
        return sanitize_value(value[0])  # Handle first element for tuples/lists
    else:
        raise ValueError(f"Value '{value}' cannot be converted to float.")


def sanitize_dataset_x(data: Any) -> List[float]:
    """Sanitize dataset.X to ensure all values are convertible to float."""
    if not hasattr(data, "X") or not isinstance(data.X, (list, tuple)):
        st.error("Dataset does not have a valid 'X' attribute.")
        return []

    sanitized_data = []
    for idx, value in enumerate(data.X):
        try:
            # Handle if value is a list or tuple
            if isinstance(value, (list, tuple)):
                sanitized_data.append(sanitize_value(value))
            else:
                sanitized_data.append(sanitize_value(value))
        except ValueError:
            st.warning(
                f"Value at index {idx} is not numeric: {value}. Replacing with 0.0."
            )
            sanitized_data.append(0.0)
    return sanitized_data


def render_train_interface(
    TrainCls: Type,
    graph: bool = True,
    hidden_layer: bool = True,
    parameter_control: bool = False,
) -> None:
    datasets_map = minitorch.datasets
    st.write("## Sandbox for Model Training")

    st.markdown("### Dataset")
    col1, col2 = st.columns(2)

    # Date selection for points
    points_date = col2.date_input("Select a date", value=date.today())

    # Ensure points_date is of type date
    if not isinstance(points_date, date):
        st.error("Please select a valid date.")
        points_date = date.today()  # Fallback to today if invalid

    # Convert selected date to number of points
    days_difference = (points_date - date.today()).days
    points = max(1, days_difference)  # Ensure points is at least 1

    selected_dataset = col1.selectbox("Select dataset", list(datasets_map.keys()))

    @st.cache
    def get_dataset(selected_dataset: str, points: int) -> Any:
        data = datasets_map[selected_dataset](points)
        return data

    raw_dataset = get_dataset(selected_dataset, points)

    # Debug print to check the type and contents of raw_dataset
    st.write(f"Raw dataset: {raw_dataset}, Type: {type(raw_dataset)}")
    st.write(f"Data.X: {getattr(raw_dataset, 'X', None)}")

    sanitized_x = sanitize_dataset_x(raw_dataset)

    fig = plots.plot_out(sanitized_x)
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig)

    st.markdown("### Model")

    hidden_layers = (
        int(
            st.number_input(
                "Size of hidden layer", min_value=1, max_value=200, step=1, value=2
            )
        )
        if hidden_layer
        else 0
    )

    train = TrainCls(hidden_layers)

    @st.cache
    def get_train(hidden_layers: int) -> str:
        one_output = train.run_one(sanitized_x[0])
        G = graph_builder.GraphBuilder().run(one_output)
        return nx.nx_pydot.to_pydot(G).to_string()

    if graph:
        graph_str = get_train(hidden_layers)
        if st.checkbox("Show Graph"):
            st.graphviz_chart(graph_str)

    if parameter_control:
        st.markdown("### Parameters")
        for n, p in train.model.named_parameters():
            value = st.slider(
                f"Parameter: {n}", min_value=-10.0, max_value=10.0, value=float(p.data)
            )
            p.update(value)

    oned = st.checkbox("Show X-Axis Only (For Simple)", False)

    def plot() -> go.Figure:
        def contour(ls: List[List[float]]) -> List[float]:
            if hasattr(train, "run_many"):
                t = train.run_many([float(x[0]) for x in ls])
                return [float(t[i, 0]) for i in range(len(ls))]
            else:
                return [
                    float(train.run_one(float(x[0])).data)
                    if hasattr(train.run_one(x[0]), "data")
                    else float(train.run_one(float(x[0])))
                    for x in ls
                ]

        fig = plots.plot_out(sanitized_x, contour, size=15, oned=oned)
        fig.update_layout(width=600, height=600)
        return fig

    st.markdown("### Initial setting")
    st.write(plot())

    # Initialize hyperparameters
    max_epochs = 500
    learning_rate = 0.01
    st_train_button = None

    if hasattr(train, "train"):
        st.markdown("### Hyperparameters")
        col1, col2 = st.columns(2)
        learning_rate = col1.selectbox(
            "Learning rate", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0], index=2
        )

        max_epochs = col2.number_input(
            "Number of epochs", min_value=1, step=25, value=max_epochs
        )

        col1, col2 = st.columns(2)
        st_train_button = col1.empty()
        col2.button("Stop Model")

    st_progress = st.empty()
    st_epoch_timer = st.empty()
    st_epoch_image = st.empty()
    st_epoch_plot = st.empty()
    st_epoch_stats = st.empty()

    start_time = time.time()

    df: List[Dict[str, Any]] = []

    def log_fn(
        epoch: int, total_loss: float, correct: int, losses: List[float]
    ) -> None:
        time_elapsed = time.time() - start_time
        if hasattr(train, "train"):
            st_progress.progress(epoch / max_epochs)
            time_per_epoch = time_elapsed / (epoch + 1)
            st_epoch_timer.markdown(
                "Epoch {}/{}. Time per epoch: {:,.3f}s. Time left: {:,.2f}s.".format(
                    epoch,
                    max_epochs,
                    time_per_epoch,
                    (max_epochs - epoch) * time_per_epoch,
                )
            )
        df.append({"epoch": epoch, "loss": total_loss, "correct": correct})
        st_epoch_stats.write(pd.DataFrame(reversed(df)))

        st_epoch_image.plotly_chart(plot())
        if hasattr(train, "train"):
            loss_graph = go.Scatter(mode="lines", x=list(range(len(losses))), y=losses)
            fig = go.Figure(loss_graph)
            fig.update_layout(
                title="Loss Graph",
                xaxis=dict(range=[0, max_epochs]),
                yaxis=dict(range=[0, max(losses)]),
            )
            st_epoch_plot.plotly_chart(fig)

            print(
                f"Epoch: {epoch}/{max_epochs}, loss: {total_loss}, correct: {correct}"
            )

    if (
        hasattr(train, "train")
        and st_train_button
        and st_train_button.button("Train Model")
    ):
        train.train(sanitized_x, learning_rate, max_epochs, log_fn)
    else:
        log_fn(0, 0, 0, [0])

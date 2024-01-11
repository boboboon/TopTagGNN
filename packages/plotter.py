"""Script to handle our different type of plotting."""
from pathlib import Path

import matplotlib.pyplot as plt


def train_history_plot(train_history: list, figure_dir: Path) -> None:
    """Plot and save a training history graph.

    This function takes a training history as a list and a directory path to save the training
    history graph. It plots the training loss and validation loss over training epochs, labels
    the axes, adds a legend, and saves the graph as "loss.png" in the specified directory.

    Args:
        train_history (list): A list containing training history data, typically obtained from
            a deep learning training process. It should include keys like "loss" and "val_loss".
        figure_dir (Path): A `Path` object specifying the directory where the training history
            graph will be saved.

    Example:
        train_history = {
            "loss": [0.2, 0.1, 0.05, ...],
            "val_loss": [0.3, 0.15, 0.08, ...]
        }
        figure_dir = Path("/path/to/figure_directory")
        train_history_plot(train_history, figure_dir)
    """
    plt.plot(train_history.history["loss"], label="Training")
    plt.plot(train_history.history["val_loss"], label="Validation")
    plt.ylabel("Cross-entropy Loss")
    plt.xlabel("Training Epoch")
    plt.legend()
    plt.savefig(figure_dir / "loss.png", dpi=300)
    plt.clf()

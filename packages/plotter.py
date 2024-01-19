"""Script to handle our different type of plotting."""
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import ndarray


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


def roc_auc_plot(tpr: ndarray, fpr: ndarray, figure_dir: Path) -> None:
    """Plots a roc_auc curve for our model's performance.

    Args:
        tpr (ndarray): Our true positive rate
        fpr (ndarray): Our false positive rate
        figure_dir (Path): Where our figures live
    """
    plt.plot(tpr, 1 / fpr)
    plt.yscale("log")
    plt.ylabel("Background rejection")
    plt.xlabel("Signal efficiency")
    plt.savefig(figure_dir / "roc.png", dpi=300)
    plt.clf()


def plot_background_rejection_vs_pt(
    br_array_dict: dict,
    pt_bins: ndarray,
    figure_dir: Path,
) -> None:
    """Plot background rejection versus jet pT for different signal efficiencies.

    Args:
        br_array_dict (dict): Dictionary containing background rejection arrays for different
          signal efficiencies.
        pt_bins (numpy.ndarray): Array defining bin edges for pT.
        figure_dir (Path): Directory to save the plot.
    """
    plot_bins = pt_bins / 1e6  # Set plot on TeV scale

    plt.figure(figsize=(8, 6))

    for efficiency, br_array in br_array_dict.items():
        label = f"$\\epsilon_{{sig}} = {efficiency}$"

        plt.step(plot_bins, br_array, "-", where="post", label=label)

    plt.ylabel("Background rejection")
    plt.xlabel("Jet pT (TeV)")
    plt.legend()
    plt.savefig(figure_dir / "br_vs_pt.png", dpi=300)
    plt.show()  # Display the plot if needed

"""Helps avoid circular imports."""
import numpy as np


class DataContainer:
    """A custom data container class to hold data, labels, and weights for training and testing.

    Args:
        data (np.array): The input data array with shape (num_samples, max_constits, num_features).
        labels (np.array): The labels array with shape (num_samples, num_classes).
        weights (np.array): The weights array with shape (num_samples,).

    Attributes:
        data (np.array): The input data array.
        labels (np.array): The labels array.
        weights (np.array): The weights array.

    Example:
        data_container = DataContainer(train_data, train_labels, train_weights)
    """

    def __init__(self, data: np.array, labels: np.array, weights: np.array) -> None:
        """Initializes a data container for training and testing data.

        Args:
            data (np.array): The input data with shape (num_samples, max_constits, num_features).
            labels (np.array): The labels with shape (num_samples, num_classes).
            weights (np.array): The sample weights with shape (num_samples,).
        """
        self.data = data
        self.labels = labels
        self.weights = weights

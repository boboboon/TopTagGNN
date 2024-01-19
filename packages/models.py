"""Handles our model creation and prepping data to feed into them."""
from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from energyflow.archs import EFN
from sklearn import metrics

from packages.container import DataContainer


def efn_model_generator() -> EFN:
    """Creates our efn model.

    Returns:
        (EFN): Our EFN model with desired parameters
    """
    return EFN(
        input_dim=2,
        Phi_sizes=(350, 350, 350, 350, 350),
        F_sizes=(300, 300, 300, 300, 300),
        Phi_k_inits="glorot_normal",
        F_k_inits="glorot_normal",
        latent_dropout=0.084,
        F_dropouts=0.036,
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=6.3e-5),
        output_dim=1,
        output_act="sigmoid",
        summary=False,
    )


def hldnn_model_generator(train_data: np.array) -> tf.keras.Sequential():
    """Builds our hldnn model.

    Args:
        train_data (_type_): Training data for the model so we can know the size #! JUST PASS SIZE!

    Returns:
        model (tf.keras.Sequential()): Our model output.
    """
    # Build hlDNN
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=train_data.shape[1:]))

    # Hidden layers
    for _ in range(5):
        model.add(
            tf.keras.layers.Dense(180, activation="relu", kernel_initializer="glorot_uniform"),
        )

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))

    # Compile hlDNN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-5),
        # from_logits set to False for uniformity with energyflow settings
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def evaluate_model(
    model: EFN | tf.keras.Sequential(),
    test_dataset: DataContainer,
    batch_size: int,
    signal_efficiencies: list,
) -> dict:
    """Evaluate classification metrics for a given model on a test dataset.

    Args:
        model (EFN | tf.keras.Sequential): The trained classification model.
        test_dataset (DataContainer): The test dataset.
        batch_size (int): Batch size for predictions.
        signal_efficiencies (list): List of different thresholds for calculating metrics.

    Returns:
        (dict): Dictionary of different metrics including predictions.
    """
    # Set up the logger
    logger = logging.getLogger(__name__)

    # Generate predictions from the model
    default_threshold = 0.5
    predictions = model.predict(test_dataset, batch_size=batch_size)[:, 0]
    discrete_predictions = (predictions > default_threshold).astype(int)

    # Evaluate metrics
    auc = metrics.roc_auc_score(test_dataset.labels, predictions)
    acc = metrics.accuracy_score(test_dataset.labels, discrete_predictions)

    # Evaluate background rejection at fixed signal efficiency working points
    fpr, tpr, thresholds = metrics.roc_curve(test_dataset.labels, predictions)

    # Define the signal efficiency levels to check
    background_rejections = []

    # Calculate background rejection for each signal efficiency level
    for signal_efficiency in signal_efficiencies:
        point_index = np.argmax(tpr > signal_efficiency)
        background_rejection = 1 / fpr[point_index]
        background_rejections.append(background_rejection)

    # Logging the results
    logger.info("\nPerformance metrics evaluated over testing set:")
    logger.info("AUC score: %f", auc)
    logger.info("ACC score: %f", acc)

    for i, signal_efficiency in enumerate(signal_efficiencies):
        logger.info(
            "Background rejection at %.1f signal efficiency: %f",
            signal_efficiency,
            background_rejections[i],
        )

    # Return the evaluation metrics and predictions as a dictionary
    return {
        "AUC": auc,
        "ACC": acc,
        "Predictions": predictions.tolist(),  # Convert predictions to a list
        "Background Rejections": {
            f"Signal Efficiency {efficiency}": br
            for efficiency, br in zip(signal_efficiencies, background_rejections)
        },
    }


def calculate_background_rejection_vs_pt(
    jet_pt: np.ndarray,
    predictions: np.ndarray,
    test_labels: np.ndarray,
    signal_efficiencies: list,
    pt_bins: np.ndarray,
) -> dict:
    """Calculate background rejection values versus jet momentum for specified thresholds.

    Args:
        jet_pt (numpy.ndarray): Array containing the pT values of jets.
        predictions (numpy.ndarray): Array of model predictions.
        test_labels (numpy.ndarray): Array of true labels for the test dataset.
        signal_efficiencies (list): List of signal efficiencies to calculate
        background rejection for.
        pt_bins (numpy.ndarray): Array defining bin edges for pT.

    Returns:
        dict: A dictionary containing background rejection values for each
        specified signal efficiency.
    """
    # Initialize a dictionary to store background rejection arrays
    br_arrays = {efficiency: np.zeros(len(pt_bins)) for efficiency in signal_efficiencies}

    # Loop through pT bins
    for i in range(len(pt_bins)):
        if i < len(pt_bins) - 1:
            # Filter jets by pT using boolean indexing
            condition = np.logical_and(jet_pt > pt_bins[i], jet_pt < pt_bins[i + 1])
            bin_indeces = np.where(condition)[0]

            # Extract predictions and labels within the current pT bin
            bin_predictions = predictions[bin_indeces]
            bin_labels = test_labels[bin_indeces]

            # Calculate the ROC curve for this pT bin
            fpr, tpr, thresholds = metrics.roc_curve(bin_labels, bin_predictions)

            # Calculate background rejection for specified signal efficiencies
            for efficiency in signal_efficiencies:
                efficiency_index = np.argmax(tpr > efficiency)
                br_arrays[efficiency][i] = 1 / fpr[efficiency_index]

    # Duplicate last entry in background rejection arrays
    for efficiency in signal_efficiencies:
        br_arrays[efficiency][-1] = br_arrays[efficiency][-2]

    return br_arrays

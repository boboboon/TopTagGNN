"""This code has been heavily adapted from Kevin Greif: https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/blob/master/preprocessing.py?ref_type=heads."""

# Numerical imports
from pathlib import Path

import h5py
import numpy as np
import sklearn as skl
import tensorflow as tf
from loguru import logger
from sklearn.model_selection import train_test_split

from packages import config_loader as cl
from packages import models


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


def prepare_data(config: cl.Config, train_path: Path(), test_path: Path()) -> tuple:
    """Prepares data and model based on the provided configuration.

    Args:
        config (config_loader.Config): Configuration for the data preparation.
        train_path (Path): Path to our training data file (HDF5 format).
        test_path (Path): Path to our testing data file (HDF5 format).

    Returns:
        tuple: A tuple containing the model and datasets for training, validation, and testing.
    """
    test = h5py.File(test_path, "r")
    train = h5py.File(train_path, "r")
    logger.info("Separating train and test data into constituent parts")

    # Load tagger configuration
    tagger_config = load_tagger_config(config)

    # Extract data vector names
    data_vector_names = train.attrs.get(tagger_config["data_vector_names"])

    if data_vector_names is None:
        error_message = "data_vector_names is None. Cannot proceed with data preparation."
        raise ValueError(error_message)

    train_dict = {key: train[key][: config.n_train_jets, ...] for key in data_vector_names}
    test_dict = {key: test[key][: config.n_test_jets, ...] for key in data_vector_names}

    # Pre-process data using the specified function
    logger.info("Preprocessing data")
    train_data = tagger_config["pre_processing_function"](train_dict)
    test_data = tagger_config["pre_processing_function"](test_dict)

    # Extract labels and weights
    train_labels = train["labels"][: config.n_train_jets]
    train_weights = train["weights"][: config.n_train_jets]
    test_labels = test["labels"][: config.n_test_jets]
    test_weights = test["weights"][: config.n_test_jets]

    # Nice containers for our data.
    logger.info("Putting data in containers")
    train_data_container = DataContainer(
        data=train_data,
        labels=train_labels,
        weights=train_weights,
    )
    test_data_container = DataContainer(
        data=test_data,
        labels=test_labels,
        weights=test_weights,
    )

    logger.info(f"Building {config.tagger_type} model")
    model = tagger_config["model_loading_function"]()

    logger.info("Preparing data For model")
    train_dataset, test_dataset, valid_dataset = tagger_config["data_loading_function"](
        config,
        train_data_container,
        test_data_container,
    )

    return model, train_dataset, valid_dataset, test_dataset


def load_tagger_config(config: cl.Config) -> dict:
    """Loads tagger information.

    Args:
        config (cl.Config): Our config file

    Raises:
        ValueError: If it doesn't recognise your config type

    Returns:
        dict: Our tagger type config dictionary
    """
    # Define a dictionary to map tagger types to corresponding data vector names,
    # preprocessing functions, and model building functions
    tagger_config = {
        "hldnn": {
            "data_vector_names": "hl",
            "pre_processing_function": high_level,
            "model_loading_function": models.hldnn_model_generator,
            "data_loading_function": prepare_hldnn_data,
        },
        "efn": {
            "data_vector_names": "constit",
            "pre_processing_function": lambda data_dict: constituent(
                data_dict,
                config.max_constits,
            ),
            "model_loading_function": models.efn_model_generator,
            "data_loading_function": prepare_efn_data,
        },
    }

    # Check if the specified tagger_type is supported``
    if config.tagger_type not in tagger_config:
        error_message = f"Unsupported tagger_type: {config.tagger_type}"
        logger.error(error_message)

    # Get configuration for the specified tagger_type
    return tagger_config[config.tagger_type]


def constituent(data_dict: dict, max_constits: int) -> np.ndarray:
    """Constituent - This function applies a standard preprocessing to the jet data.

    Args:
        data_dict (dict): The python dictionary containing all of
    the constituent level quantities. Standard naming conventions will be
    assumed.
        max_constits (int): The maximum number of constituents to consider in
    preprocessing. Cut jet constituents at this number.


    Returns:
        np.ndarray: The seven constituent level quantities, stacked along the last
    axis.
    """
    ############################## Load Data ###################################

    # Pull data from data dict
    pt = data_dict["fjet_clus_pt"][:, :max_constits]
    eta = data_dict["fjet_clus_eta"][:, :max_constits]
    phi = data_dict["fjet_clus_phi"][:, :max_constits]
    energy = data_dict["fjet_clus_E"][:, :max_constits]

    # Find location of zero pt entries in each jet
    mask = pt == 0

    ########################## Angular Coordinates #############################

    # 1. Center hardest constituent in eta/phi plane. First find eta and
    # phi shifts to be applied
    eta_shift = eta[:, 0]
    phi_shift = phi[:, 0]

    # Apply them using np.newaxis
    eta_center = eta - eta_shift[:, np.newaxis]
    phi_center = phi - phi_shift[:, np.newaxis]

    # Fix discontinuity in phi at +/- pi using np.where
    phi_center = np.where(phi_center > np.pi, phi_center - 2 * np.pi, phi_center)
    phi_center = np.where(phi_center < -np.pi, phi_center + 2 * np.pi, phi_center)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    second_eta = eta_center[:, 1]
    second_phi = phi_center[:, 1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi / 2
    eta_rot = eta_center * np.cos(alpha[:, np.newaxis]) + phi_center * np.sin(alpha[:, np.newaxis])
    phi_rot = -eta_center * np.sin(alpha[:, np.newaxis]) + phi_center * np.cos(alpha[:, np.newaxis])

    # 3. If needed, reflect so 3rd hardest constituent is in positive eta
    third_eta = eta_rot[:, 2]
    parity = np.where(third_eta < 0, -1, 1)
    eta_flip = (eta_rot * parity[:, np.newaxis]).astype(np.float32)
    # Cast to float32 needed to keep numpy from turning eta to double precision

    # 4. Calculate R with pre-processed eta/phi
    radius = np.sqrt(eta_flip**2 + phi_rot**2)

    ############################# pT and Energy ################################

    # Set zero elements to 1 before taking the log
    log_pt = np.log(np.where(mask, 1, pt))
    log_energy = np.log(np.where(mask, 1, energy))

    # Sum pt and energy in each jet
    sum_pt = np.sum(pt, axis=1)
    sum_energy = np.sum(energy, axis=1)

    # Normalize pt and energy and again take logarithm
    lognorm_pt = np.log(
        np.where(
            mask,
            1,
            pt / sum_pt[:, np.newaxis],
        ),  # Set zero elements to 1 before taking the log
    )
    lognorm_energy = np.log(np.where(mask, 1, energy / sum_energy[:, np.newaxis]))

    ########################### Finalize and Return ############################

    # Reset all of the original zero entries to zero
    eta_flip = np.where(mask, 0, eta_flip)
    phi_rot = np.where(mask, 0, phi_rot)
    radius = np.where(mask, 0, radius)

    # Stack along last axis
    features = [eta_flip, phi_rot, log_pt, log_energy, lognorm_pt, lognorm_energy, radius]
    return np.stack(features, axis=-1)


def high_level(data_dict: dict) -> np.ndarray:
    """High_level - This function "standardizes" each of the high level quantities.

    Args:
        data_dict (dict): The python dictionary containing all of
    the high level quantities. No naming conventions assumed.

    Returns:
        np.ndarray: The high level quantities, stacked along the last dimension.
    """
    # Empty list to accept pre-processed high level quantities
    features = []

    scale1_limit = 1e5
    scale2_limit = 1e11
    scale3_limit = 1e17

    for quant in data_dict.values():
        if scale1_limit < quant.max() <= scale2_limit:
            quant /= 1e6
        elif scale2_limit < quant.max() <= scale3_limit:
            quant /= 1e12
        elif quant.max() > scale3_limit:
            quant /= 1e18

        # Calculated mean and standard deviation
        mean = quant.mean()
        stddev = quant.std()

        # Standardize and append to list
        standard_quant = (quant - mean) / stddev
        features.append(standard_quant)

    # Stack quantities and return
    return np.stack(features, axis=-1)


def create_tf_dataset(data_list: list, batch_size: int) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a list of data and batch it.

    This function takes a list of data and creates a TensorFlow dataset using
    `tf.data.Dataset.from_tensor_slices`. It then batches the dataset into
    smaller batches specified by the `batch_size` parameter.

    Args:
        data_list (list): A list or tuple of data to be converted into a TensorFlow dataset.
        batch_size (int): The size of batches to create from the data.

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing the data batched by the
        specified `batch_size`.

    Example:
        data_list = [data1, data2, data3]
        batch_size = 32
        dataset = create_tf_dataset(data_list, batch_size)
    """
    return tf.data.Dataset.from_tensor_slices(data_list).batch(batch_size)


def prepare_efn_data(
    config: cl.Config,
    train_data_container: DataContainer,
    test_data_container: DataContainer,
) -> tuple:
    """Prepares the data for our efn model.

    Args:
        config (cl.Config): Configuration object for the EFN model.
        train_data_container (DataContainer): Data container for training data.
        test_data_container (DataContainer): Data container for testing data.

    Returns:
        tuple: Datasets required for training, validation, and testing.
    """
    train_data = train_data_container.data
    train_labels = train_data_container.labels
    train_weights = train_data_container.weights

    test_data = test_data_container.data
    test_labels = test_data_container.labels
    test_weights = test_data_container.weights

    # Extract relevant columns for angular and pt data
    angular_indices = slice(0, 2)
    pt_index = 2

    train_angular = train_data[:, :, angular_indices]
    train_pt = train_data[:, :, pt_index]

    test_angular = test_data[:, :, angular_indices]
    test_pt = test_data[:, :, pt_index]

    # Split the training data into train and validation sets

    (
        train_angular,
        valid_angular,
        train_pt,
        valid_pt,
        train_labels,
        valid_labels,
        train_weights,
        valid_weights,
    ) = train_test_split(
        train_angular,
        train_pt,
        train_labels,
        train_weights,
        test_size=config.valid_fraction,
    )

    batch_size = config.batch_size

    # Build tensorflow data sets
    train_list = [train_pt, train_angular, train_labels, train_weights]
    train_sets = tuple(
        [tf.data.Dataset.from_tensor_slices(i).batch(batch_size) for i in train_list],
    )
    train_data = tf.data.Dataset.zip(train_sets[:2])
    train_dataset = tf.data.Dataset.zip((train_data,) + train_sets[2:])

    valid_list = [valid_pt, valid_angular, valid_labels, valid_weights]
    valid_sets = tuple(
        [tf.data.Dataset.from_tensor_slices(i).batch(batch_size) for i in valid_list],
    )
    valid_data = tf.data.Dataset.zip(valid_sets[:2])
    valid_dataset = tf.data.Dataset.zip((valid_data,) + valid_sets[2:])

    test_list = [test_pt, test_angular, test_labels, test_weights]
    test_sets = tuple([tf.data.Dataset.from_tensor_slices(i).batch(batch_size) for i in test_list])
    test_data = tf.data.Dataset.zip(test_sets[:2])
    test_dataset = tf.data.Dataset.zip((test_data,) + test_sets[2:])

    return train_dataset, valid_dataset, test_dataset


def prepare_hldnn_data(
    config: cl.Config,
    train_data_container: DataContainer,
    test_data_container: DataContainer,
) -> tuple:
    """Prepares the data for the hldnn model.

    Args:
        config (cl.Config): Configuration object for the hldnn model.
        train_data_container (DataContainer): Data container for training data.
        test_data_container (DataContainer): Data container for testing data.

    Returns:
        tuple: Datasets required for training, validation, and testing.
    """
    train_data = train_data_container.data
    train_labels = train_data_container.labels
    train_weights = train_data_container.weights

    test_data = test_data_container.data
    test_labels = test_data_container.labels
    test_weights = test_data_container.weights

    # Make train / valid split using sklearn train_test_split function
    (
        train_data,
        valid_data,
        train_labels,
        valid_labels,
        train_weights,
        valid_weights,
    ) = skl.model_selection.train_test_split(
        train_data,
        train_labels,
        train_weights,
        test_size=config.valid_fraction,
    )

    batch_size = config.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data, train_labels, train_weights),
    ).batch(batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_data, valid_labels, valid_weights),
    ).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data, test_labels, test_weights),
    ).batch(batch_size)

    return (train_dataset, valid_dataset, test_dataset)

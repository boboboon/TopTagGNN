"""What's the deal with tagging?"""
from pathlib import Path

import h5py
from loguru import logger

import packages.config_loader as cl
from packages import data_loading, models


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
            "pre_processing_function": data_loading.high_level,
        },
        "efn": {
            "data_vector_names": "constit",
            "pre_processing_function": lambda data_dict: data_loading.constituent(
                data_dict,
                config.max_constits,
            ),
        },
    }

    # Check if the specified tagger_type is supported``
    if config.tagger_type not in tagger_config:
        error_message = f"Unsupported tagger_type: {config.tagger_type}"
        logger.error(error_message)

    # Get configuration for the specified tagger_type
    return tagger_config[config.tagger_type]


def prepare_data(config: cl.Config, train: h5py.File, test: h5py.File) -> tuple:
    """Prepares data and model based on the provided configuration.

    Args:
        config (cl.Config): Configuration for the data preparation.
        train (h5py.File): Training data file (HDF5 format).
        test (h5py.File): Testing data file (HDF5 format).

    Returns:
        tuple: A tuple containing the model and datasets for training, validation, and testing.
    """
    # Extract labels and weights
    train_labels = train["labels"][: config.n_train_jets]
    train_weights = train["weights"][: config.n_train_jets]
    test_labels = test["labels"][: config.n_test_jets]
    test_weights = test["weights"][: config.n_test_jets]

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
    train_data = tagger_config["pre_processing_function"](train_dict)
    test_data = tagger_config["pre_processing_function"](test_dict)

    # Determine the tagger type
    tagger_type = config.tagger_type

    if tagger_type == "efn":
        # Build and compile EFN model
        model = models.efn_model_generator()
        train_dataset, test_dataset, valid_dataset = data_loading.prepare_efn_data(
            train_data,
            test_data,
            train_labels,
            test_labels,
            train_weights,
            test_weights,
        )
    else:
        # Build model for other tagger types
        model = models.hldnn_model_generator(train_data)

        # Prepare tensorflow datasets
        train_dataset, valid_dataset, test_dataset = data_loading.prepare_hldnn_data(
            train_data,
            test_data,
            train_labels,
            test_labels,
            train_weights,
            test_weights,
        )

    return model, train_dataset, valid_dataset, test_dataset


def main(config: cl.Config) -> None:
    """Performs our jet tagging.

    Args:
        config (cl.Config): Our config file.
    """
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"

    test = h5py.File(test_path, "r")
    train = h5py.File(test_path, "r")

    (model, train_dataset, valid_dataset, test_dataset) = prepare_data(
        config,
        train,
        test,
    )


if __name__ == "__main__":
    config = cl._arg_config()
    main(config)

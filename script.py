"""What's the deal with tagging?"""
from pathlib import Path

import h5py
from loguru import logger

from packages import config_loader, data_loading, models


def prepare_data(config: config_loader.Config, train: h5py.File, test: h5py.File) -> tuple:
    """Prepares data and model based on the provided configuration.

    Args:
        config (config_loader.Config): Configuration for the data preparation.
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
    tagger_config = data_loading.load_tagger_config(config)

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

    # Nice container for our data.
    train_data_container = data_loading.DataContainer(
        data=train_data,
        labels=train_labels,
        weights=train_weights,
    )
    test_data_container = data_loading.DataContainer(
        data=test_data,
        labels=test_labels,
        weights=test_weights,
    )

    if tagger_type == "efn":
        # Build and compile EFN model
        model = models.efn_model_generator()
        train_dataset, test_dataset, valid_dataset = data_loading.prepare_efn_data(
            config,
            train_data_container,
            test_data_container,
        )
    else:
        # Build model for other tagger types
        model = models.hldnn_model_generator(train_data)

        # Prepare tensorflow datasets
        train_dataset, valid_dataset, test_dataset = data_loading.prepare_hldnn_data(
            config,
            train_data_container,
            test_data_container,
        )

    return model, train_dataset, valid_dataset, test_dataset


def main(config: config_loader.Config) -> None:
    """Performs our jet tagging.

    Args:
        config (config_loader.Config): Our config file.
    """
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"

    test = h5py.File(test_path, "r")
    train = h5py.File(test_path, "r")

    (model, train_dataset, valid_dataset, test_dataset) = prepare_data(config, train, test)

    logger.info("Wahey!")


if __name__ == "__main__":
    config = config_loader._arg_config()
    main(config)

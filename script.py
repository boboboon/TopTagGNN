"""What's the deal with tagging?"""
from pathlib import Path

import h5py
import numpy as np
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


def prepare_data(
    config: cl.Config,
    train_data: np.array,
    test_data: np.array,
    train_labels: np.array,
    test_labels: np.array,
    train_weights: np.array,
    test_weights: np.array,
):
    """Prepares our data depending on our tagger config.

    Args:
        config (cl.Config): Our config file
        train_data (np.array): File as named
        test_data (np.array): File as named
        train_labels (np.array): File as named
        test_labels (np.array): File as named
        train_weights (np.array): File as named
        test_weights (np.array): File as named

    Returns:
        (tuple): A tuple of our model and datasets for training
    """
    tagger_type = config.tagger_type
    if tagger_type == "efn":
        # Build and compile EFN
        model = models.efn_model_generator()
        (train_dataset, test_dataset, valid_dataset) = data_loading.prepare_efn_data(
            train_data,
            test_data,
            train_labels,
            test_labels,
            train_weights,
            test_weights,
        )
        return (model, train_dataset, valid_dataset, test_dataset)

    # For all other models, data sets can be built using the same process, so
    # these are handled together

    model = models.hldnn_model_generator(train_data)

    # Build tensorflow datasets.
    # In tf.keras' "fit" API, the first argument is the inputs, the second is
    # the labels, and the third is an optional "sample weight". This is where
    # the training weights should be applied.
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    (train_dataset, valid_dataset, test_dataset) = data_loading.prepare_hldnn_data(
        train_data,
        test_data,
        train_labels,
        test_labels,
        train_weights,
        test_weights,
    )

    return (model, train_dataset, valid_dataset, test_dataset)


def main(config: cl.Config) -> None:
    """Performs our jet tagging.

    Args:
        config (cl.Config): Our config file.
    """
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"

    test = h5py.File(test_path, "r")
    train = h5py.File(test_path, "r")

    tagger_config = load_tagger_config(config)

    data_vector_names = train.attrs.get(tagger_config["data_vector_names"])

    train_dict = {key: train[key][: config.n_train_jets, ...] for key in data_vector_names}  # type: ignore
    test_dict = {key: test[key][: config.n_test_jets, ...] for key in data_vector_names}  # type: ignore

    train_data = tagger_config["pre_processing_function"](train_dict)
    test_data = tagger_config["pre_processing_function"](test_dict)
    num_data_features = train_data.shape[-1]

    train_labels = train["labels"][: config.n_train_jets]
    train_weights = train["weights"][: config.n_train_jets]
    test_labels = test["labels"][: config.n_test_jets]
    test_weights = test["labels"][: config.n_test_jets]

    jet_pt = test["fjet_pt"][: config.n_test_jets]


if __name__ == "__main__":
    config = cl._arg_config()
    main(config)

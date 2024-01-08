"""What's the deal with tagging?"""
from pathlib import Path

import h5py
from loguru import logger

import packages.config_loader as cl
from packages import pre_processing


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
            "pre_processing_function": pre_processing.high_level,
        },
        "efn": {
            "data_vector_names": "constit",
            "pre_processing_function": lambda data_dict: pre_processing.constituent(
                data_dict, config.max_constits
            ),
        },
    }

    # Check if the specified tagger_type is supported``
    if config.tagger_type not in tagger_config:
        error_message = f"Unsupported tagger_type: {config.tagger_type}"
        logger.error(error_message)

    # Get configuration for the specified tagger_type
    return tagger_config[config.tagger_type]


def main(config):
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"

    test = h5py.File(test_path, "r")
    train = h5py.File(test_path, "r")

    tagger_config = load_tagger_config(config)

    data_vector_names = train.attrs.get(tagger_config["data_vector_names"])

    train_dict = {key: train[key][: config.n_train_jets, ...] for key in data_vector_names}  # type: ignore
    test_dict = {key: test[key][: config.n_test_jets, ...] for key in data_vector_names}  # type: ignore


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)  # Convert the string to a Path object
    config = load_config(config_path)  # Pass the Path object to the load_config function
    main(config)

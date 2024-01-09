"""Our config loader package."""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from packages import data_loading, models


@dataclass
class Config:
    """Our config class so we know the data types of what we load in."""

    n_train_jets: int
    n_test_jets: int
    valid_fraction: float
    max_constits: int
    tagger_type: str
    num_epochs: int
    batch_size: int
    data_path: Path
    figure_path: Path


def load_config(config_path: Path) -> Config:
    """Load in our yml config file.

    Args:
        config_path (Path): Path to our config file

    Returns:
        Config: Our config file
    """
    if not config_path.is_file():
        error_message = f"Config file '{config_path}' not found."
        logger.error(error_message)
        raise ValueError(error_message)

    with Path.open(config_path) as config_file:
        config_data = yaml.safe_load(config_file)

    return Config(**config_data)


def _arg_config() -> Config:
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=Path, help="path to config file", required=True)
    args = parser.parse_args()
    return load_config(args.config_file)


def load_tagger_config(config: Config) -> dict:
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
            "model_loading_function": models.hldnn_model_generator,
        },
        "efn": {
            "data_vector_names": "constit",
            "pre_processing_function": lambda data_dict: data_loading.constituent(
                data_dict,
                config.max_constits,
            ),
            "model_loading_function": models.efn_model_generator,
        },
    }

    # Check if the specified tagger_type is supported``
    if config.tagger_type not in tagger_config:
        error_message = f"Unsupported tagger_type: {config.tagger_type}"
        logger.error(error_message)

    # Get configuration for the specified tagger_type
    return tagger_config[config.tagger_type]

"""Our config loader package."""
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger


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

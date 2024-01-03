import argparse
import yaml
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    n_train_jets: int
    n_test_jets: int
    valid_fraction: float
    max_constits: int
    tagger_type: str
    num_epochs: int
    batch_size: int
    data_path: Path
    figure_path: Path

def load_config(config_path):
    if not config_path.is_file():
        print(f"Config file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)

    return Config(**config_data)

def parse_args():
    parser = argparse.ArgumentParser(description="Your Script Description")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the config file")
    return parser.parse_args()



import argparse
import yaml
from pathlib import Path
import sys

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_path):
    if not config_path.is_file():
        print(f"Config file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)

    return Config(config_data)

def parse_args():
    parser = argparse.ArgumentParser(description="Your Script Description")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the config file")
    return parser.parse_args()

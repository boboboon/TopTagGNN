from packages.config_loader import load_config, parse_args
from pathlib import Path
import h5py


def main(config):
    # Access configuration settings like config.valid_fraction
    valid_fraction = config.valid_fraction
    n_train_jets = config.n_train_jets

    # Continue with your code
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"

    test = h5py.File(test_path, 'r')
    print(n_train_jets)
    # Now you can use config.valid_fraction and other config settings in your script

if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)  # Convert the string to a Path object
    config = load_config(config_path)  # Pass the Path object to the load_config function
    main(config)


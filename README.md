# TopTagGNN

## Introduction

TopTagGNN is a PyTorch-based approach to the ATLAS Top Tagging Open Data Set.

## Getting Started

The existing Top Tagging code from CERN ([ATLAS-top-tagging-open-data](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/tree/master)) doesn't have particularly good segmentation or documentation to get it up and running. Here are some tips to help you get started:

### Python Version

- Tensorflow doesn't work well with the most recent Python versions (e.g., Python 3.11 at the time of writing this). However, Python 3.7.12 seems to work fine.

### Managing Python Versions

- We recommend using `pyenv` and `pyenv-virtualenv` to manage your Python versions. If you're not already using them, consider setting up `pyenv` for a smoother experience.

### Virtual Environment

- Create a virtual environment of the required Python version and select it as your interpreter, especially if you are working from inside VSCode.

### Energyflow Compatibility

- Note that Energyflow may not be compatible with certain Python versions, as it hasn't been updated in 2 years. However, it is still a very interesting library to work with.

### Configuration Files

- For explanations on how the config files work, you can refer to the `config loader` section.

## Usage

- Have fun testing your TopTagGNN model! To get started, please follow these steps:

1. Look at the `requirements.txt` file and download the required modules into your nice `pyenv` virtual environment.

2. Start working on your TopTagGNN project with the specified Python version and dependencies.

## Contributing

Feel free to contribute to this project by submitting pull requests or reporting issues.

## License

This project is licensed under the [MIT License](LICENSE).

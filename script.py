"""What's the deal with tagging?"""
from pathlib import Path

import h5py
import numpy as np
import sklearn as skl
import tensorflow as tf
from energyflow.archs import EFN
from loguru import logger

import packages.config_loader as cl
from packages import data_loading


def efn_model_generator() -> EFN():
    """Creates our efn model.

    Returns:
        (EFN): Our EFN model with desired parameters
    """
    return EFN(
        input_dim=2,
        Phi_sizes=(350, 350, 350, 350, 350),
        F_sizes=(300, 300, 300, 300, 300),
        Phi_k_inits="glorot_normal",
        F_k_inits="glorot_normal",
        latent_dropout=0.084,
        F_dropouts=0.036,
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=6.3e-5),
        output_dim=1,
        output_act="sigmoid",
        summary=False,
    )


def prepare_efn_data(
    train_data: np.array,
    test_data: np.array,
    train_labels: np.array,
    test_labels: np.array,
    train_weights: np.array,
    test_weights: np.array,
) -> tuple:
    """Prepares the data for our efn model.

    Args:
        train_data (np.array): As named.
        test_data (np.array): As named.
        train_labels (np.array): As named.
        test_labels (np.array): As named.
        train_weights (np.array): As named.
        test_weights (np.array): As named.

    Returns:
        (tuple): Our datasets required for training
    """
    # For EFN, take only eta, phi, and log(pT) quantities, and package into
    # a single dataset. We want each element of the data set to have shape:
    #   ((batch_size, max_constits, 1), (batch_size, max_constits, 2))  # noqa: ERA001
    # We can do this using tensorflow Dataset's "zip" function.
    # This code assumes quantities are ordered (eta, phi, pT, ...)
    train_angular = train_data[:, :, 0:2]
    train_pt = train_data[:, :, 2]

    test_angular = test_data[:, :, 0:2]
    test_pt = test_data[:, :, 2]

    # Make train / valid split using sklearn train_test_split function
    (
        train_angular,
        valid_angular,
        train_pt,
        valid_pt,
        train_labels,
        valid_labels,
        train_weights,
        valid_weights,
    ) = skl.model_selection.train_test_split(
        train_angular,
        train_pt,
        train_labels,
        train_weights,
        test_size=config.valid_fraction,
    )

    batch_size = config.batch_size
    # Build tensorflow data sets
    train_list = [train_pt, train_angular, train_labels, train_weights]
    train_sets = tuple(
        [tf.data.Dataset.from_tensor_slices(i).batch(batch_size) for i in train_list]
    )
    train_data = tf.data.Dataset.zip(train_sets[:2])
    train_dataset = tf.data.Dataset.zip((train_data,) + train_sets[2:])

    valid_list = [valid_pt, valid_angular, valid_labels, valid_weights]
    valid_sets = tuple(
        [tf.data.Dataset.from_tensor_slices(i).batch(batch_size) for i in valid_list]
    )
    valid_data = tf.data.Dataset.zip(valid_sets[:2])
    valid_dataset = tf.data.Dataset.zip((valid_data,) + valid_sets[2:])

    test_list = [test_pt, test_angular, test_labels, test_weights]
    test_sets = tuple([tf.data.Dataset.from_tensor_slices(i).batch(batch_size) for i in test_list])
    test_data = tf.data.Dataset.zip(test_sets[:2])
    test_dataset = tf.data.Dataset.zip((test_data,) + test_sets[2:])

    return (train_dataset, valid_dataset, test_dataset)


def hldnn_model_generator(train_data: np.array) -> tf.keras.Sequential():
    """Builds our hldnn model.

    Args:
        train_data (_type_): Training data for the model so we can know the size #! JUST PASS SIZE!

    Returns:
        model (tf.keras.Sequential()): Our model output.
    """
    # Build hlDNN
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=train_data.shape[1:]))

    # Hidden layers
    for _ in range(5):
        model.add(
            tf.keras.layers.Dense(180, activation="relu", kernel_initializer="glorot_uniform")
        )

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))

    # Compile hlDNN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-5),
        # from_logits set to False for uniformity with energyflow settings
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


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
        model = efn_model_generator()
        (train_dataset, test_dataset, valid_dataset) = prepare_efn_data(
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

    model = hldnn_model_generator(train_data)

    # Make train / valid split using sklearn train_test_split function
    (
        train_data,
        valid_data,
        train_labels,
        valid_labels,
        train_weights,
        valid_weights,
    ) = skl.model_selection.train_test_split(
        train_data,
        train_labels,
        train_weights,
        test_size=config.valid_fraction,
    )

    # Build tensorflow datasets.
    # In tf.keras' "fit" API, the first argument is the inputs, the second is
    # the labels, and the third is an optional "sample weight". This is where
    # the training weights should be applied.
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    batch_size = config.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data, train_labels, train_weights),
    ).batch(batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_data, valid_labels, valid_weights),
    ).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data, test_labels, test_weights),
    ).batch(batch_size)

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

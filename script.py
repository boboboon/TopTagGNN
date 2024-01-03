from pathlib import Path

import h5py
import tensorflow as tf
from energyflow.archs import EFN
from sklearn.model_selection import train_test_split

from packages import pre_processing
from packages.config_loader import load_config, parse_args


def model_generator(tagger_type, train_data_shape):
    if tagger_type == "efn":
        model = EFN(
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
        return model

    if tagger_type == "hldnn":
        # Build hlDNN
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=train_data_shape))

        # Hidden layers
        for _ in range(5):
            model.add(
                tf.keras.layers.Dense(180, activation="relu", kernel_initializer="glorot_uniform")
            )

        # Output layer
        model.add(
            tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform")
        )

        # Compile hlDNN
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=4e-5),
            # from_logits set to False for uniformity with energyflow settings
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        )
        return model


def model_data_loader(config, train, test):
    train_labels = train["labels"][: config.n_train_jets]
    train_weights = train["weights"][: config.n_train_jets]
    test_labels = test["labels"][: config.n_test_jets]
    test_weights = test["labels"][: config.n_test_jets]

    tagger_config = load_tagger_config(config)
    data_vector_names = test.attrs.get(tagger_config["data_vector_names"])
    train_dict = {key: train[key][: config.n_train_jets, ...] for key in data_vector_names}
    test_dict = {key: test[key][: config.n_test_jets, ...] for key in data_vector_names}

    train_data = tagger_config["pre_processing_function"](train_dict)
    test_data = tagger_config["pre_processing_function"](test_dict)

    if config.tagger_type == "efn":
        # For EFN, take only eta, phi, and log(pT) quantities, and package into
        # a single dataset. We want each element of the data set to have shape:
        #   ((batch_size, max_constits, 1), (batch_size, max_constits, 2))
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
        ) = train_test_split(
            train_angular, train_pt, train_labels, train_weights, test_size=config.valid_fraction
        )

        # Build tensorflow data sets
        train_list = [train_pt, train_angular, train_labels, train_weights]
        train_sets = tuple(
            [tf.data.Dataset.from_tensor_slices(i).batch(config.batch_size) for i in train_list]
        )
        train_data = tf.data.Dataset.zip(train_sets[:2])
        train_dataset = tf.data.Dataset.zip((train_data,) + train_sets[2:])

        valid_list = [valid_pt, valid_angular, valid_labels, valid_weights]
        valid_sets = tuple(
            [tf.data.Dataset.from_tensor_slices(i).batch(config.batch_size) for i in valid_list]
        )
        valid_data = tf.data.Dataset.zip(valid_sets[:2])
        valid_dataset = tf.data.Dataset.zip((valid_data,) + valid_sets[2:])

        test_list = [test_pt, test_angular, test_labels, test_weights]
        test_sets = tuple(
            [tf.data.Dataset.from_tensor_slices(i).batch(config.batch_size) for i in test_list]
        )
        test_data = tf.data.Dataset.zip(test_sets[:2])
        test_dataset = tf.data.Dataset.zip((test_data,) + test_sets[2:])

        return train_dataset, valid_dataset, test_dataset

    # Make train / valid split using sklearn train_test_split function
    (
        train_data,
        valid_data,
        train_labels,
        valid_labels,
        train_weights,
        valid_weights,
    ) = train_test_split(train_data, train_labels, train_weights, test_size=config.valid_fraction)

    # Build tensorflow datasets.
    # In tf.keras' "fit" API, the first argument is the inputs, the second is
    # the labels, and the third is an optional "sample weight". This is where
    # the training weights should be applied.
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data, train_labels, train_weights)
    ).batch(config.batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_data, valid_labels, valid_weights)
    ).batch(config.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels, test_weights)).batch(
        config.batch_size
    )

    return train_dataset, valid_dataset, test_dataset


def load_tagger_config(config):
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
        raise ValueError(f"Unsupported tagger_type: {config.tagger_type}")

    # Get configuration for the specified tagger_type
    tagger_type_config = tagger_config[config.tagger_type]

    return tagger_type_config


def main(config):
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"
    train_path = data_path / "train_public.h5"

    test = h5py.File(test_path, "r")
    train = h5py.File(train_path, "r")
    jet_pt = test["fjet_pt"][: config.n_test_jets]
    train_data_shape = 5

    model = model_generator(config.tagger_type, train_data_shape)
    train_dataset, valid_dataset, test_dataset = model_data_loader(config, train, test)


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)  # Convert the string to a Path object
    config = load_config(config_path)  # Pass the Path object to the load_config function
    main(config)

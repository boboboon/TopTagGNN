from packages.config_loader import load_config, parse_args
from packages import pre_processing
from pathlib import Path
import h5py
from energyflow.archs import EFN
import tensorflow as tf


def build_efn_model(config):
    # Build and compile EFN
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
        output_act='sigmoid',
        summary=False
    )

    return model

def build_hldnn_model(config):
    # Build hlDNN
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(config.max_constits, config.num_data_features)))

    # Hidden layers
    for _ in range(5):
        model.add(tf.keras.layers.Dense(
            180,
            activation='relu',
            kernel_initializer='glorot_uniform')
        )

    # Output layer
    model.add(tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer='glorot_uniform')
    )

    # Compile hlDNN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    )

    return model

def load_tagger_config(config):
    # Define a dictionary to map tagger types to corresponding data vector names,
    # preprocessing functions, and model building functions
    tagger_config = {
        'hldnn': {
            'data_vector_names': 'hl',
            'pre_processing_function': pre_processing.high_level,
            'model_builder': build_hldnn_model,
            'data_shape': (config.n_jets, 15),
        },
        'efn': {
            'data_vector_names': 'constit',
            'pre_processing_function': lambda data_dict: pre_processing.constituent(data_dict, config.max_constits),
            'model_builder': build_efn_model,
            'data_shape': (config.n_jets, config.max_constits, 7),
        },
    }

    # Check if the specified tagger_type is supported
    if config.tagger_type not in tagger_config:
        raise ValueError(f"Unsupported tagger_type: {config.tagger_type}")

    # Get configuration for the specified tagger_type
    tagger_type_config = tagger_config[config.tagger_type]

    return tagger_type_config

def main(config):
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"
    test = h5py.File(test_path, 'r')

    tagger_config=load_tagger_config(config)

    data_vector_names = test.attrs.get(tagger_config["data_vector_names"])


    test_dict = {key: config.test[key][:config.n_test_jets, ...] for key in data_vector_names}

    test_data = tagger_config["pre_processing_function"](test_dict)
    num_data_features = test_data.shape[-1]

    test_labels = test['labels'][:config.n_test_jets]
    test_weights = test['labels'][:config.n_test_jets]
    jet_pt = test['fjet_pt'][:config.n_test_jets]



    

    
if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)  # Convert the string to a Path object
    config = load_config(config_path)  # Pass the Path object to the load_config function
    main(config)


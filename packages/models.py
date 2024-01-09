"""Handles our model creation and prepping data to feed into them."""
import numpy as np
import tensorflow as tf
from energyflow.archs import EFN


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
            tf.keras.layers.Dense(180, activation="relu", kernel_initializer="glorot_uniform"),
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

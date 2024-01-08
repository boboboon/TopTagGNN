"""This code has been heavily adapted from Kevin Greif: https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/blob/master/preprocessing.py?ref_type=heads."""

# Numerical imports
import numpy as np
import sklearn as skl
import tensorflow as tf

import packages.config_loader as cl


def constituent(data_dict: dict, max_constits: int) -> np.ndarray:
    """Constituent - This function applies a standard preprocessing to the jet data.

    Args:
        data_dict (dict): The python dictionary containing all of
    the constituent level quantities. Standard naming conventions will be
    assumed.
        max_constits (int): The maximum number of constituents to consider in
    preprocessing. Cut jet constituents at this number.


    Returns:
        np.ndarray: The seven constituent level quantities, stacked along the last
    axis.
    """
    ############################## Load Data ###################################

    # Pull data from data dict
    pt = data_dict["fjet_clus_pt"][:, :max_constits]
    eta = data_dict["fjet_clus_eta"][:, :max_constits]
    phi = data_dict["fjet_clus_phi"][:, :max_constits]
    energy = data_dict["fjet_clus_E"][:, :max_constits]

    # Find location of zero pt entries in each jet. This will be used as a
    # mask to re-zero out entries after all preprocessing steps
    mask = np.asarray(pt == 0).nonzero()

    ########################## Angular Coordinates #############################

    # 1. Center hardest constituent in eta/phi plane. First find eta and
    # phi shifts to be applied
    eta_shift = eta[:, 0]
    phi_shift = phi[:, 0]

    # Apply them using np.newaxis
    eta_center = eta - eta_shift[:, np.newaxis]
    phi_center = phi - phi_shift[:, np.newaxis]

    # Fix discontinuity in phi at +/- pi using np.where
    phi_center = np.where(phi_center > np.pi, phi_center - 2 * np.pi, phi_center)
    phi_center = np.where(phi_center < -np.pi, phi_center + 2 * np.pi, phi_center)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    second_eta = eta_center[:, 1]
    second_phi = phi_center[:, 1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi / 2
    eta_rot = eta_center * np.cos(alpha[:, np.newaxis]) + phi_center * np.sin(alpha[:, np.newaxis])
    phi_rot = -eta_center * np.sin(alpha[:, np.newaxis]) + phi_center * np.cos(alpha[:, np.newaxis])

    # 3. If needed, reflect so 3rd hardest constituent is in positive eta
    third_eta = eta_rot[:, 2]
    parity = np.where(third_eta < 0, -1, 1)
    eta_flip = (eta_rot * parity[:, np.newaxis]).astype(np.float32)
    # Cast to float32 needed to keep numpy from turning eta to double precision

    # 4. Calculate R with pre-processed eta/phi
    radius = np.sqrt(eta_flip**2 + phi_rot**2)

    ############################# pT and Energy ################################

    # Take the logarithm, ignoring -infs which will be set to zero later
    log_pt = np.log(pt)
    log_energy = np.log(energy)

    # Sum pt and energy in each jet
    sum_pt = np.sum(pt, axis=1)
    sum_energy = np.sum(energy, axis=1)

    # Normalize pt and energy and again take logarithm
    lognorm_pt = np.log(pt / sum_pt[:, np.newaxis])
    lognorm_energy = np.log(energy / sum_energy[:, np.newaxis])

    ########################### Finalize and Return ############################

    # Reset all of the original zero entries to zero
    eta_flip[mask] = 0
    phi_rot[mask] = 0
    log_pt[mask] = 0
    log_energy[mask] = 0
    lognorm_pt[mask] = 0
    lognorm_energy[mask] = 0
    radius[mask] = 0

    # Stack along last axis
    features = [eta_flip, phi_rot, log_pt, log_energy, lognorm_pt, lognorm_energy, radius]
    return np.stack(features, axis=-1)


def high_level(data_dict: dict) -> np.ndarray:
    """High_level - This function "standardizes" each of the high level quantities.

    Args:
        data_dict (dict): The python dictionary containing all of
    the high level quantities. No naming conventions assumed.

    Returns:
        np.ndarray: The high level quantities, stacked along the last dimension.
    """
    # Empty list to accept pre-processed high level quantities
    features = []

    scale1_limit = 1e5
    scale2_limit = 1e11
    scale3_limit = 1e17

    for quant in data_dict.values():
        if scale1_limit < quant.max() <= scale2_limit:
            quant /= 1e6
        elif scale2_limit < quant.max() <= scale3_limit:
            quant /= 1e12
        elif quant.max() > scale3_limit:
            quant /= 1e18

        # Calculated mean and standard deviation
        mean = quant.mean()
        stddev = quant.std()

        # Standardize and append to list
        standard_quant = (quant - mean) / stddev
        features.append(standard_quant)

    # Stack quantities and return
    return np.stack(features, axis=-1)


def prepare_efn_data(
    config: cl.Config,
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


def prepare_hldnn_data(
    config: cl.Config,
    train_data: np.array,
    test_data: np.array,
    train_labels: np.array,
    test_labels: np.array,
    train_weights: np.array,
    test_weights: np.array,
) -> tuple:
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

    return (train_dataset, valid_dataset, test_dataset)

"""What's the deal with tagging?"""
from pathlib import Path

from loguru import logger
from numpy import linspace

from packages import config_loader, data_loading, models, plotter


def main(config: config_loader.Config) -> None:
    """Performs our jet tagging.

    Args:
        config (config_loader.Config): Our config file.
    """
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"
    logger.add("log_files/info.log", level="INFO", mode="w")

    logger.info("Building models and formatting data")

    model, jet_pt, train_data_container, test_data_container = data_loading.prepare_data(
        config=config,
        train_path=test_path,
        test_path=test_path,
    )
    logger.info("Preparing data For model")
    tagger_config = data_loading.load_tagger_config(config.tagger_type)
    data_loading_function = tagger_config["data_loading_function"]
    train_dataset, test_dataset, valid_dataset = data_loading_function(
        config,
        train_data_container,
        test_data_container,
    )

    logger.info("Training Model")

    batch_size = config.batch_size

    train_history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        batch_size=batch_size,
        epochs=config.num_epochs,
        verbose=1,
    )
    plotter.train_history_plot(train_history, Path(config.figure_path))
    models.evaluate_model(model, test_dataset, config.batch_size)

    # Evaluate our model
    signal_efficiencies = [0.3, 0.5, 0.8]
    model_eval_dict = models.evaluate_model(
        model,
        test_dataset,
        config.batch_size,
        signal_efficiencies,
    )
    # Finally make a plot of the background rejection versus jet pT. Start by making
    # a set of pT bins and empty vectors to accept B.R. values. Note pt bin array
    # defines bin edges

    pt_bins = linspace(350000, 3150000, 15)

    br_array_dict = models.calculate_background_rejection_vs_pt(
        jet_pt,
        model_eval_dict["predictions"],
        test_data_container.labels,
        signal_efficiencies,
        pt_bins,
    )
    plotter.plot_background_rejection_vs_pt(
        br_array_dict,
        pt_bins,
        config.figure_dir,
    )


if __name__ == "__main__":
    config = config_loader._arg_config()
    logger.info("Loaded config with the following parameters:")
    for param_name, param_value in vars(config).items():
        logger.info(f"{param_name}: {param_value}")
    main(config)

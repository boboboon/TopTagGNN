"""What's the deal with tagging?"""
from pathlib import Path

from loguru import logger

from packages import config_loader, data_loading, plotter


def main(config: config_loader.Config) -> None:
    """Performs our jet tagging.

    Args:
        config (config_loader.Config): Our config file.
    """
    data_path = Path("/Users/lucascurtin/Desktop/CERN_DATA")
    test_path = data_path / "test.h5"
    logger.add("log_files/info.log", level="INFO", mode="w")

    logger.info("Build models and formatting data")

    (model, train_dataset, valid_dataset, test_dataset) = data_loading.prepare_data(
        config=config,
        train_path=test_path,
        test_path=test_path,
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"],
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
    plotter.train_history_plot(train_history, config.figure_dir)


if __name__ == "__main__":
    config = config_loader._arg_config()
    logger.info("Loaded config with the following parameters:")
    for param_name, param_value in vars(config).items():
        logger.info(f"{param_name}: {param_value}")
    main(config)

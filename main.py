import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
import logging
import subprocess

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]

logger = logging.getLogger(__name__)

# This automatically reads in the configuration
@hydra.main(config_name="config", version_base="1.1", config_path=".")
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            logger.info("Starting the download step.")
            sample = config["etl"]["sample"]
            command = [
                "python", "components/get_data/manual.py",
                "--sample", sample,
                "--artifact_name", "sample.csv",
                "--artifact_type", "raw_data",
                "--artifact_description", "Raw file as downloaded"
            ]

            # Run the command using subprocess
            subprocess.run(command, check=True)
    
        if "basic_cleaning" in active_steps:
            logger.info("Starting the basic cleaning step.")
            basic_cleaning_path = os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning")
            
            _ = mlflow.run(
                basic_cleaning_path,
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )
    
        if "data_check" in active_steps:
            logger.info("Starting the data check step.")
            data_check_path = os.path.join(hydra.utils.get_original_cwd(), "src", "data_check")
            
            _ = mlflow.run(
                data_check_path,
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                'main',
                parameters = {
                    "input": "clean_sample.csv:latest", 
                    "test_size": config["modeling"]["test_size"], 
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )
        
        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH
        
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"
                }
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                'main',
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )

if __name__ == "__main__":
    go()

"""Preprocess the NSD metadata so that the COCO metadata can be added to it."""

from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from nsd_access import NSDAccess
from omegaconf import DictConfig, OmegaConf

load_dotenv()


@hydra.main(config_path="../../configs/preprocessing", config_name="preprocessing")
def preprocess_nsd_coco(cfg: DictConfig) -> None:
    """
    Preprocess the NSD-COCO dataset.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra found in the
        configs/preprocessing/preprocessing.yaml file.
    """
    logger.info(f"Configuration: {cfg}")

    try:
        large_data_path = Path(cfg.data.large_data_path)
    except OmegaConf.MissingMandatoryValue:
        logger.error("LARGE_DATA_PATH is not set. Please set it using one of the following methods:")
        logger.error("1. Set the LARGE_DATA_PATH environment variable in the .env file.")
        logger.error(
            "2. Provide it as a command-line argument: python script.py data.large_data_path=/path/to/large/data"
        )
        return

    nsd_dir = large_data_path / cfg.data.nsd_directory

    logger.info(f"Preprocessing NSD-COCO data from: {nsd_dir}")

    # TODO: Implement preprocessing steps here
    # For example
    # - Load the data

    # Load the metadata file using nsd_access
    nsd = NSDAccess(nsd_dir)

    nsd_stim_desc = pd.read_csv(nsd.stimuli_description_file, index_col=0)

    # TODO either add the code or the preprocessed COCO metadata file

    # - Clean the data
    # - Transform the data
    # - Save the processed data

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess_nsd_coco()

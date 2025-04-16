"""Preprocess the NSD metadata so that the COCO metadata can be added to it."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from nsd_access import NSDAccess
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
load_dotenv()


@hydra.main(config_path="../../../configs/data", config_name="data")
def preprocess_nsd_coco(cfg: DictConfig) -> None:
    """
    Preprocess the NSD-COCO dataset.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra found in the
        configs/data/data.yaml file.
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

    # Load the metadata file using nsd_access
    nsd = NSDAccess(nsd_dir)
    nsd_stim_desc = pd.read_csv(nsd.stimuli_description_file, index_col=0)
    nsd_coco_ids = set(nsd_stim_desc["cocoId"].tolist())

    # Load the VG metadata
    vg_metadata = load_dataset(
        cfg.data.vg_metadata_hf_identifier,
        cache_dir=cfg.data.cache_dir,
        split="train",
    )
    # Convert to pandas DataFrame
    vg_df = vg_metadata.to_pandas()
    # Drop duplicates based on 'cocoid'
    vg_df_unique = vg_df.drop_duplicates(subset=["cocoid"], keep="first")
    # Rename the 'cocoid' column to 'cocoId'
    vg_df_unique = vg_df_unique.rename(columns={"cocoid": "cocoId"})
    # Get the COCO IDs from the VG metadata
    vg_coco_ids = set(vg_df_unique["cocoId"])

    # Find the overlap between the NSD and VG COCO IDs
    overlap_coco_ids = nsd_coco_ids & vg_coco_ids
    logger.info(f"Found {len(overlap_coco_ids)} overlapping COCO IDs.")

    # Merge the NSD and VG metadata on the COCO ID, make sure there are no extra indices
    nsd_vg_metadata = nsd_stim_desc.merge(vg_df_unique, on="cocoId", how="inner")

    # Create a new subdirectory in the NSD directory
    nsd_vg_dir = nsd_dir / "nsd_vg"
    nsd_vg_dir.mkdir(parents=True, exist_ok=True)

    # Save the merged metadata to a CSV file
    nsd_vg_metadata.to_csv(nsd_vg_dir / "nsd_vg_metadata.csv", index=False)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess_nsd_coco()

"""Compute image embeddings for NSD data using CLIP-like models."""

import logging
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore")


@hydra.main(config_path="../../configs/model", config_name="image_embeddings")
def run_image_embeddings(cfg: DictConfig) -> None:
    """
    Pre-compute image embeddings for CLIP-like models. The embeddings serve as
    input features for the neural encoding analysis.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    # Load the COCO image dataset
    coco_dataset = load_dataset(
        cfg.huggingface.dataset_name,
        cache_dir=cfg.data.dataset_cache_dir,
        split="train",
    )

    # Check if the output directory exists
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally filter or select based on NSD metadata
    nsd_dir = Path(cfg.data.large_data_path) / cfg.data.nsd_directory
    nsd_vg_metadata = pd.read_csv(nsd_dir / "nsd_vg" / "nsd_vg_metadata.csv")
    coco_dataset = coco_dataset.select(list(nsd_vg_metadata["nsdId"].values))

    # For each model in config
    for model_id in cfg.huggingface.model_ids:
        # Check if the embeddings already exist
        output_path = Path(cfg.data.output_dir) / f"{model_id.replace('/', '_')}_embeddings.npy"
        if output_path.exists():
            logger.info(f"Embeddings for {model_id} already exist at {output_path}. Skipping.")
            continue

        logger.info(f"Loading {model_id}")

        # Load CLIP model and processor
        processor = CLIPProcessor.from_pretrained(
            model_id,
            cache_dir=cfg.data.model_cache_dir,
        )
        model = CLIPVisionModel.from_pretrained(
            model_id,
            cache_dir=cfg.data.model_cache_dir,
        )
        model.eval()

        # Initialize list to store embeddings
        batch_embeddings = []
        batch_size = cfg.batch_size if "batch_size" in cfg else 8
        device = torch.device("cuda" if cfg.device == "cuda" else "cpu")

        # Move model to device
        model.to(device)

        # Iterate through the dataset, compute embeddings in batches
        for i in tqdm(range(0, len(coco_dataset), batch_size), desc=f"Computing embeddings for {model_id}"):
            batch = coco_dataset[i : i + batch_size]["image"]  # noqa
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs).pooler_output
            if cfg.device == "cuda":
                outputs = outputs.cpu()
            batch_embeddings.append(outputs.numpy())

        # Concatenate and save
        embeddings = np.concatenate(batch_embeddings, axis=0)
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    run_image_embeddings()

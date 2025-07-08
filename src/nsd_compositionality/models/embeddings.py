"""Compute image embeddings for NSD data using CLIP-like models."""

import logging
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, CLIPProcessor, ViTImageProcessor, ViTModel

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore")


@hydra.main(config_path="../../../configs/model", config_name="embeddings")
def run_embeddings(cfg: DictConfig) -> None:
    """
    Pre-compute embeddings for CLIP-like models. The embeddings serve as
    input features for the neural encoding analysis.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    # Load the COCO image dataset
    coco_image_dataset = load_dataset(
        cfg.huggingface.image_dataset_name,
        cache_dir=cfg.data.dataset_cache_dir,
        split=cfg.data.image_split,
    )
    coco_text_dataset = load_dataset(
        cfg.huggingface.text_dataset_name,
        cache_dir=cfg.data.dataset_cache_dir,
        split=cfg.data.text_split,
    )

    # Check if the output directory exists
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter or select based on NSD metadata
    nsd_dir = Path(cfg.data.large_data_path) / cfg.data.nsd_directory
    nsd_vg_metadata = pd.read_csv(nsd_dir / "nsd_vg" / "nsd_vg_metadata.csv")
    coco_image_dataset = coco_image_dataset.select(list(nsd_vg_metadata["nsdId"].values))
    # Stack/concatenate the cocoId from the NSD metadata onto the dataset
    coco_image_dataset = coco_image_dataset.add_column("cocoid", nsd_vg_metadata["cocoId"].values)
    # Merge all features from the text dataset onto the image dataset using cocoid
    coco_text_df = coco_text_dataset.to_pandas()
    # Only keep one random caption per image if specified
    if cfg.data.keep_one_caption:
        coco_text_df = coco_text_df.drop_duplicates(subset=["cocoid"])
    # Merge the text dataset with the image dataset using pandas
    coco_image_df = coco_image_dataset.to_pandas()
    # Select the text_df based on the cocoid in the image_df
    coco_text_df = coco_text_df[coco_text_df["cocoid"].isin(coco_image_df["cocoid"])]
    # Put the coco_text_df in the same order as the coco_image_df based on the cocoid
    coco_text_df = coco_text_df.set_index("cocoid").reindex(coco_image_df["cocoid"]).reset_index()
    # Remove the cocoid column from the text dataset
    coco_text_df = coco_text_df.drop(columns=["cocoid"])
    # Convert back to a huggingface dataset
    coco_text_dataset = Dataset.from_pandas(coco_text_df)
    # And concatenate the image and text datasets
    coco_image_dataset = concatenate_datasets([coco_image_dataset, coco_text_dataset], axis=1)

    # For each model in config
    for model_id in cfg.huggingface.model_ids:
        # Check if the embeddings already exist
        output_path = Path(cfg.data.output_dir) / f"{model_id.replace('/', '_')}_embeddings.npy"
        if output_path.exists() and not cfg.override:
            logger.info(f"Embeddings for {model_id} already exist at {output_path}. Skipping.")
            continue

        logger.info(f"Loading {model_id}")

        # Load model and processor
        if "graphormer" in model_id.lower() or "clip" in model_id.lower():
            processor = CLIPProcessor.from_pretrained(
                model_id,
                cache_dir=cfg.data.model_cache_dir,
                trust_remote_code=True,
            )
            model = AutoModel.from_pretrained(
                model_id,
                cache_dir=cfg.data.model_cache_dir,
                trust_remote_code=True,
            )
            num_hidden_layers = model.config.vision_config.num_hidden_layers
        else:
            processor = ViTImageProcessor.from_pretrained(
                model_id,
                cache_dir=cfg.data.model_cache_dir,
            )
            model = ViTModel.from_pretrained(
                model_id,
                cache_dir=cfg.data.model_cache_dir,
                trust_remote_code=True,
            )
            num_hidden_layers = model.config.num_hidden_layers
        model.eval()

        # Initialize list to store embeddings
        batch_embeddings = []
        layer_wise_embeddings: dict = {f"layer_{i}": [] for i in range(num_hidden_layers)}
        batch_size = cfg.batch_size if "batch_size" in cfg else 8
        device = torch.device("cuda" if cfg.device == "cuda" else "cpu")

        # Move model to device
        model.to(device)

        # Iterate through the dataset, compute embeddings in batches
        for i in tqdm(range(0, len(coco_image_dataset), batch_size), desc=f"Computing embeddings for {model_id}"):
            # Use both text and image embeddings and graph embeddings if available
            if "graphormer" in model_id.lower():
                # TODO fix
                text_batch = coco_image_dataset[i : i + batch_size]["text"]
                image_batch = coco_image_dataset[i : i + batch_size]["image"]
                graph_batch = coco_image_dataset[i : i + batch_size][cfg.data.graph_column]
                inputs = processor(
                    images=image_batch,
                    text=text_batch,
                    return_tensors="pt",
                )
                inputs["graph_input"] = graph_batch
            elif "clip" in model_id.lower():
                text_batch = coco_image_dataset[i : i + batch_size]["text"]  # noqa
                image_batch = coco_image_dataset[i : i + batch_size]["image"]  # noqa
                inputs = processor(images=image_batch, text=text_batch, return_tensors="pt")
            else:
                image_batch = coco_image_dataset[i : i + batch_size]["image"]
                inputs = processor(images=image_batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if cfg.by_layer:
                    # Skip the first initial embedding output [1:]
                    outputs = model(**inputs, output_hidden_states=True).hidden_states[1:]
                    if cfg.device == "cuda":
                        outputs = tuple([layer.cpu() for layer in outputs])
                    for i, layer in enumerate(outputs):
                        layer_wise_embeddings[f"layer_{i}"].append(layer.mean(axis=1).numpy())
                else:
                    outputs = model(**inputs).pooler_output
                    if cfg.device == "cuda":
                        outputs = outputs.cpu()
                    batch_embeddings.append(outputs.numpy())

        # Concatenate and save
        if cfg.by_layer:
            # Save each layer individually
            for layer, embeddings in layer_wise_embeddings.items():
                embeddings = np.concatenate(embeddings, axis=0)
                output_path = Path(cfg.data.output_dir) / f"{model_id.replace('/', '_')}_{layer}_embeddings.npy"
                np.save(output_path, embeddings)
        else:
            embeddings = np.concatenate(batch_embeddings, axis=0)
            np.save(output_path, embeddings)

        logger.info(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    run_embeddings()

"""Comprehensive evaluation of CLIP and related models on compositional understanding benchmarks."""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from nsd_compositionality.models.graph_clip_model.modeling_graph_clip import GraphCLIPModel

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore")


class WinogradEvaluator:
    """Evaluator for Winoground dataset."""

    def __init__(self, model: CLIPModel, processor: CLIPProcessor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score between an image and text."""
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # For GraphCLIP models, explicitly pass graph_input=None
            if hasattr(self.model, "graph_model"):
                outputs = self.model(**inputs, graph_input=None)
                logits_per_image = outputs.logits_image_text
            else:
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
            return logits_per_image[0, 0].cpu().item()

    def evaluate_example(self, example: Dict) -> Dict[str, Union[float, bool, str]]:
        """
        Evaluate a single Winoground example.

        Winoground contains pairs of images (image_0, image_1) and captions (caption_0, caption_1)
        where each caption should match its corresponding image better than the other image.
        """
        image_0 = example["image_0"]
        image_1 = example["image_1"]
        caption_0 = example["caption_0"]
        caption_1 = example["caption_1"]

        # Compute all four similarity scores
        score_i0_c0 = self.compute_similarity(image_0, caption_0)
        score_i0_c1 = self.compute_similarity(image_0, caption_1)
        score_i1_c0 = self.compute_similarity(image_1, caption_0)
        score_i1_c1 = self.compute_similarity(image_1, caption_1)

        # Text accuracy: captions should prefer their corresponding images
        text_correct_0 = score_i0_c0 > score_i1_c0
        text_correct_1 = score_i1_c1 > score_i0_c1
        text_correct = text_correct_0 and text_correct_1

        # Image accuracy: images should prefer their corresponding captions
        image_correct_0 = score_i0_c0 > score_i0_c1
        image_correct_1 = score_i1_c1 > score_i1_c0
        image_correct = image_correct_0 and image_correct_1

        # Group accuracy: both text and image accuracy must be correct
        group_correct = text_correct and image_correct

        return {
            "text_correct": text_correct,
            "image_correct": image_correct,
            "group_correct": group_correct,
        }


class SVOProbesEvaluator:
    """Evaluator for SVO (Subject-Verb-Object) probes dataset."""

    def __init__(self, model: CLIPModel, processor: CLIPProcessor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score between an image and text."""
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # For GraphCLIP models, explicitly pass graph_input=None
            if hasattr(self.model, "graph_model"):
                outputs = self.model(**inputs, graph_input=None)
                logits_per_image = outputs.logits_image_text
            else:
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
            return logits_per_image[0, 0].cpu().item()

    def evaluate_example(self, example: Dict) -> Dict[str, Union[float, bool, str]]:
        """
        Evaluate a single SVO probes example.

        SVO probes test whether models understand compositional structure by
        evaluating similarity between images and captions with correct vs.
        incorrect subject-verb-object relationships.
        """
        pos_image = example["pos_image"]
        neg_image = example["neg_image"]
        text = example["sentence"]
        if example["subj_neg"]:
            category = "subj_neg"
        elif example["obj_neg"]:
            category = "obj_neg"
        else:
            category = "verb_neg"

        # Compute similarity scores
        score_correct = self.compute_similarity(pos_image, text)
        score_incorrect = self.compute_similarity(neg_image, text)

        # Correct if correct caption has higher similarity
        correct = score_correct > score_incorrect

        return {
            "score_correct": score_correct,
            "score_incorrect": score_incorrect,
            "correct": correct,
            "category": category,
        }


class SugarCREPEEvaluator:
    """Evaluator for SugarCREPE dataset."""

    def __init__(self, model: CLIPModel, processor: CLIPProcessor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score between an image and text."""
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # For GraphCLIP models, explicitly pass graph_input=None
            if hasattr(self.model, "graph_model"):
                outputs = self.model(**inputs, graph_input=None)
                logits_per_image = outputs.logits_image_text
            else:
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
            return logits_per_image[0, 0].cpu().item()

    def evaluate_example(self, example: Dict) -> Dict[str, Union[float, bool, str]]:
        """
        Evaluate a single SugarCREPE example.

        SugarCREPE tests compositional reasoning with positive and negative captions
        where models should prefer the compositionally correct caption.
        """
        image = example["images"][0]
        caption_positive = example["positive_caption"][0]
        caption_negative = example["negative_caption"][0]
        category = example["original_file_name"]

        # Compute similarity scores
        score_positive = self.compute_similarity(image, caption_positive)
        score_negative = self.compute_similarity(image, caption_negative)

        # Correct if positive caption has higher similarity
        correct = score_positive > score_negative

        return {
            "score_positive": score_positive,
            "score_negative": score_negative,
            "correct": correct,
            "category": category,
        }


def evaluate_dataset(
    model_id: str,
    dataset_name: str,
    dataset: Union[Dataset, List[Dict]],
    device: str = "cuda",
    cache_dir: str = None,  # type: ignore
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Evaluate a model on a specific dataset.

    Args:
        model_id: Hugging Face model identifier
        dataset_name: Name of the dataset being evaluated
        dataset: Dataset to evaluate on
        device: Device to run inference on
        cache_dir: Cache directory for models

    Returns:
        Tuple of (detailed results per example, summary metrics)
    """
    logger.info(f"Loading model {model_id}")

    # Load model and processor
    if "graphormer" in model_id.lower():
        processor = CLIPProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        model = GraphCLIPModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
    elif "clip" in model_id.lower():
        processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

    model.to(device)
    model.eval()

    # Choose appropriate evaluator based on dataset
    if "winoground" in dataset_name.lower():
        evaluator = WinogradEvaluator(model, processor, device)
    elif "svo" in dataset_name.lower():
        evaluator = SVOProbesEvaluator(model, processor, device)  # type: ignore
    elif "sugarcrepe" in dataset_name.lower():
        evaluator = SugarCREPEEvaluator(model, processor, device)  # type: ignore
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    results = []
    logger.info(f"Evaluating {len(dataset)} examples on {dataset_name}")

    for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {model_id} on {dataset_name}")):
        try:
            result = evaluator.evaluate_example(example)
            result["example_id"] = i
            result["model_id"] = model_id
            result["dataset"] = dataset_name
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate example {i}: {e}")
            continue

    # Calculate summary metrics based on dataset type
    if results:
        if dataset_name.lower() == "winoground":
            summary = {
                "model_id": model_id,
                "dataset": dataset_name,
                "text_accuracy": np.mean([r["text_correct"] for r in results]),
                "image_accuracy": np.mean([r["image_correct"] for r in results]),
                "group_accuracy": np.mean([r["group_correct"] for r in results]),
                "num_examples": len(results),
                "num_total": len(dataset),
            }
        else:  # SugarCREPE or SVO Probes
            # Get scores per category
            categories = set(r["category"] for r in results)
            category_scores = {cat: 0.0 for cat in categories}
            summary = {
                "model_id": model_id,
                "dataset": dataset_name,
                "accuracy": 0.0,
                "num_examples": len(results),
                "num_total": len(dataset),
            }
            for cat in categories:
                cat_results = [r for r in results if r["category"] == cat]
                if cat_results:
                    cat_accuracy = np.mean([r["correct"] for r in cat_results])
                    category_scores[cat] = cat_accuracy
                    summary[cat + "_accuracy"] = cat_accuracy  # type: ignore
                    summary["accuracy"] += cat_accuracy  # type: ignore
            summary["accuracy"] /= len(categories)

    else:
        if dataset_name.lower() == "winoground":
            summary = {
                "model_id": model_id,
                "dataset": dataset_name,
                "text_accuracy": 0.0,
                "image_accuracy": 0.0,
                "group_accuracy": 0.0,
                "num_examples": 0,
                "num_total": len(dataset),
            }
        else:
            summary = {
                "model_id": model_id,
                "dataset": dataset_name,
                "accuracy": 0.0,
                "num_examples": 0,
                "num_total": len(dataset),
            }

    return results, summary


@hydra.main(config_path="../../../configs/model", config_name="compositional_benchmarking")
def run_compositional_benchmarking(cfg: DictConfig) -> None:
    """
    Run comprehensive compositional benchmarking evaluation.

    This script evaluates models on compositional understanding benchmarks including:
    - Winoground: Visual-linguistic compositional reasoning
    - SVO Probes: Subject-Verb-Object compositional structure understanding
    - SugarCREPE: Compositional reasoning with positive/negative captions

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    logger.info("Starting compositional benchmarking evaluation")

    # Create output directory
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results storage
    all_results = []
    summary_results = []

    # Evaluate each dataset
    for dataset_config in cfg.datasets:
        dataset_name = dataset_config.get("name", "winoground")
        logger.info(f"Loading {dataset_name} dataset")

        try:
            # Load dataset
            hf_name = dataset_config.get("hf_name", "facebook/winoground")
            dataset_config_name = dataset_config.get("dataset_config", None)

            dataset = load_dataset(
                hf_name,
                dataset_config_name,
                cache_dir=cfg.data.dataset_cache_dir,
                split=dataset_config.get("split", "test"),
                trust_remote_code=True,
            )

            logger.info(f"Loaded {len(dataset)} examples from {dataset_name}")

            # Limit dataset size if specified
            max_examples = dataset_config.get("max_examples", 0)
            if max_examples > 0:
                dataset = dataset.select(range(min(max_examples, len(dataset))))
                logger.info(f"Limited {dataset_name} to {len(dataset)} examples")

        except Exception as e:
            logger.error(f"Failed to load {dataset_name} dataset: {e}")
            continue

        # Evaluate each model on this dataset
        for model_id in cfg.model_ids:
            logger.info(f"Evaluating {model_id} on {dataset_name}")

            # Check if results already exist
            model_output_file = output_dir / f"{model_id.replace('/', '_')}_{dataset_name}_results.csv"
            if model_output_file.exists() and not cfg.override:
                logger.info(f"Results for {model_id} on {dataset_name} already exist, skipping.")
                continue

            try:
                # Evaluate model on dataset
                model_results, model_summary = evaluate_dataset(
                    model_id=model_id,
                    dataset_name=dataset_name,
                    dataset=dataset,
                    device=cfg.device,
                    cache_dir=cfg.data.model_cache_dir,
                )

                # Store results
                all_results.extend(model_results)
                summary_results.append(model_summary)

                # Save individual model-dataset results
                if model_results:
                    model_df = pd.DataFrame(model_results)
                    model_df.to_csv(model_output_file, index=False)
                    logger.info(f"Saved results for {model_id} on {dataset_name} to {model_output_file}")

                # Log summary metrics
                logger.info(f"Results for {model_id} on {dataset_name}:")
                logger.info(f"Summary: {model_summary}")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_id} on {dataset_name}: {e}")
                continue

    # Save combined results
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_path = output_dir / "compositional_benchmarking_detailed_results.csv"
        all_results_df.to_csv(all_results_path, index=False)
        logger.info(f"Saved detailed results to {all_results_path}")

    # Save summary results
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_path = output_dir / "compositional_benchmarking_summary_results.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary results to {summary_path}")

        # Also save legacy Winoground-specific files for backward compatibility
        winoground_summary = summary_df[summary_df["dataset"] == "winoground"]
        if not winoground_summary.empty:
            winoground_summary_path = output_dir / "winoground_summary_results.csv"
            winoground_summary.to_csv(winoground_summary_path, index=False)
            logger.info(f"Saved Winoground summary results to {winoground_summary_path}")

        # Create combined LaTeX table with all datasets
        logger.info("Creating combined LaTeX table...")
        latex_data = []

        for dataset in summary_df["dataset"].unique():
            dataset_results = summary_df[summary_df["dataset"] == dataset]

            if dataset.lower() == "winoground":
                # Add Winoground metrics
                for metric in ["text_accuracy", "image_accuracy", "group_accuracy"]:
                    row = {
                        "Dataset": dataset.title(),
                        "Metric": metric.replace("_", " ").title(),
                    }
                    for _, result_row in dataset_results.iterrows():
                        model_name = result_row["model_id"].split("/")[-1]  # Use short model name
                        row[model_name] = f"{result_row[metric]:.3f}"
                    latex_data.append(row)
            else:
                # For SVO and SugarCREPE datasets - get all category-specific accuracies
                first_row = dataset_results.iloc[0]

                # Find all category-specific accuracy columns
                category_columns = [col for col in first_row.index if col.endswith("_accuracy") and col != "accuracy"]

                # Add overall accuracy first
                row = {"Dataset": dataset.title(), "Metric": "Overall"}
                for _, result_row in dataset_results.iterrows():
                    model_name = result_row["model_id"].split("/")[-1]
                    row[model_name] = f"{result_row['accuracy']:.3f}"
                latex_data.append(row)

                # Add category-specific accuracies
                for category_col in sorted(category_columns):
                    category_name = category_col.replace("_accuracy", "").replace("_", " ").title()
                    row = {"Dataset": dataset.title(), "Metric": category_name}
                    for _, result_row in dataset_results.iterrows():
                        model_name = result_row["model_id"].split("/")[-1]
                        if pd.notna(result_row[category_col]):
                            row[model_name] = f"{result_row[category_col]:.3f}"
                        else:
                            row[model_name] = "N/A"
                    latex_data.append(row)

        latex_df = pd.DataFrame(latex_data)
        latex_path = output_dir / "latex_table.csv"
        latex_df.to_csv(latex_path, index=False)
        logger.info(f"Saved combined LaTeX table to {latex_path}")

        # Print final summary table
        logger.info("Final Summary")

        for dataset in summary_df["dataset"].unique():
            logger.info(f"\n{dataset.upper()}:")
            dataset_results = summary_df[summary_df["dataset"] == dataset]

            if dataset.lower() == "winoground":
                for _, row in dataset_results.iterrows():
                    logger.info(
                        f"  {row['model_id']:<40} | "
                        f"Text: {row['text_accuracy']:.3f} | "
                        f"Image: {row['image_accuracy']:.3f} | "
                        f"Group: {row['group_accuracy']:.3f}"
                    )
            else:
                for _, row in dataset_results.iterrows():
                    logger.info(f"  {row['model_id']:<40} | Accuracy: {row['accuracy']:.3f}")
                    for col in row.index:  # type: ignore
                        if col.endswith("_accuracy") and col != "accuracy":
                            logger.info(f"    {col.replace('_', ' ').title()}: {row[col]:.3f}")

    logger.info("\nCompositional benchmarking evaluation complete!")


if __name__ == "__main__":
    run_compositional_benchmarking()

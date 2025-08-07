"""
Visual Structural Probe implementation adapted from Hewitt and Manning (2019).

This module implements structural probes that predict visual scene graph properties
from CLIP patch embeddings. Instead of syntactic distances between tokens,
we probe for spatial and semantic relationships between objects in visual scenes
using their corresponding CLIP patch representations.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from omegaconf import DictConfig
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import CLIPVisionModel

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore")


class VisualStructuralProbe(nn.Module):
    """
    Structural probe for predicting visual scene graph distances from CLIP patch embeddings.

    Adapted from Hewitt and Manning (2019) to work with visual object representations
    extracted from CLIP patch embeddings corresponding to object locations.
    """

    def __init__(self, representation_dim: int, probe_rank: int = 0):
        """
        Initialize the visual structural probe.

        Args:
            representation_dim (int): Dimension of input CLIP patch representations
            probe_rank (int): Rank of the transformation matrix. If 0, uses full rank.
        """
        super().__init__()
        self.representation_dim = representation_dim
        self.probe_rank = probe_rank if probe_rank > 0 else representation_dim

        # Linear transformation to map patch embeddings to distance space
        self.linear_transform = nn.Linear(representation_dim, self.probe_rank)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear_transform.weight)
        nn.init.constant_(self.linear_transform.bias, 0.0)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Transform patch embeddings through the probe.

        Args:
            patch_embeddings (torch.Tensor): Patch embeddings of shape (batch_size, num_objects, dim)

        Returns:
            torch.Tensor: Transformed representations of shape (batch_size, num_objects, probe_rank)
        """
        return self.linear_transform(patch_embeddings)

    def compute_distance_matrix(self, transformed_patches: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between transformed patch embeddings.

        Args:
            transformed_patches (torch.Tensor): Transformed patch embeddings
                                              of shape (batch_size, num_objects, probe_rank)

        Returns:
            torch.Tensor: Distance matrix of shape (batch_size, num_objects, num_objects)
        """
        # Compute squared L2 distances between all pairs of objects
        # Using the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T * b

        # Compute squared norms
        squared_norms = torch.sum(transformed_patches**2, dim=-1, keepdim=True)  # (batch, n_objects, 1)

        # Compute dot products
        dot_products = torch.bmm(
            transformed_patches, transformed_patches.transpose(-1, -2)
        )  # (batch, n_objects, n_objects)

        # Compute squared distances
        squared_distances = squared_norms + squared_norms.transpose(-1, -2) - 2 * dot_products

        # Take square root and ensure non-negative
        distances = torch.sqrt(torch.clamp(squared_distances, min=0.0))

        return distances


class SceneGraphComplexityProbe(nn.Module):
    """
    Probe that predicts scene graph complexity measures from image representations.

    This probe learns to predict scalar properties like:
    - Number of objects
    - Number of relationships
    - Graph depth
    """

    def __init__(self, representation_dim: int, num_properties: int = 3):
        """
        Initialize the scene graph complexity probe.

        Args:
            representation_dim (int): Dimension of input representations
            num_properties (int): Number of graph properties to predict
        """
        super().__init__()
        self.hidden_dim = representation_dim // 2

        self.predictor = nn.Sequential(
            nn.Linear(representation_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_properties),
        )

        # Initialize weights
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complexity probe.

        Args:
            representations (torch.Tensor): Input representations of shape (batch_size, dim)

        Returns:
            torch.Tensor: Predicted properties of shape (batch_size, num_properties)
        """
        return self.predictor(representations)


def extract_scene_graph_properties(graph_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract numerical properties from scene graph data for probing.

    Args:
        graph_data (Dict[str, Any]): Scene graph data containing edge_index and num_nodes

    Returns:
        Dict[str, float]: Dictionary of graph properties
    """
    if not graph_data or graph_data.get("num_nodes", 0) == 0:
        return {
            "num_nodes": 0.0,
            "num_edges": 0.0,
            "depth": 0.0,
        }

    num_nodes = float(graph_data["num_nodes"])
    edge_index = graph_data.get("edge_index", [[], []])

    if not edge_index or len(edge_index) != 2 or len(edge_index[0]) == 0:
        num_edges = 0.0
        depth = 0.0
    else:
        num_edges = float(len(edge_index[0]))
        # Compute depth (longest shortest path)
        if num_nodes > 1:
            adjacency_matrix = np.zeros((int(num_nodes), int(num_nodes)))
            for source, target in zip(edge_index[0], edge_index[1]):
                if 0 <= source < num_nodes and 0 <= target < num_nodes:
                    adjacency_matrix[int(source), int(target)] = 1
                    adjacency_matrix[int(target), int(source)] = 1

            distances = compute_tree_distances_from_adjacency(adjacency_matrix)
            finite_distances = distances[distances < float("inf")]
            depth = float(np.max(finite_distances)) if len(finite_distances) > 0 else 0.0
        else:
            depth = 0.0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "depth": depth,
    }


def compute_tree_distances_from_adjacency(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Compute tree distances using Floyd-Warshall algorithm.

    Args:
        adjacency_matrix (np.ndarray): Binary adjacency matrix representing tree structure

    Returns:
        np.ndarray: Matrix of shortest path distances
    """
    n = adjacency_matrix.shape[0]
    distances = adjacency_matrix.astype(float)

    # Initialize with infinity for non-connected nodes
    distances[distances == 0] = np.inf
    np.fill_diagonal(distances, 0)

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])

    # Replace infinity with a large value for unreachable nodes
    distances[np.isinf(distances)] = n

    return distances


def train_structural_probe(
    probe: VisualStructuralProbe,
    representations: torch.Tensor,
    target_distances: torch.Tensor,
    masks: torch.Tensor,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
    device: str = "cuda",
) -> List[float]:
    """
    Train the structural probe using MSE loss on tree distances.

    Args:
        probe (VisualStructuralProbe): The probe to train
        representations (torch.Tensor): Patch representations of shape (batch, num_objects, dim)
        target_distances (torch.Tensor): Target tree distances
        masks (torch.Tensor): Masks for valid positions
        learning_rate (float): Learning rate for optimization
        num_epochs (int): Number of training epochs
        device (str): Device to use for training

    Returns:
        List[float]: Training losses
    """
    probe.to(device)
    representations = representations.to(device)
    target_distances = target_distances.to(device)
    masks = masks.to(device)

    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    losses = []

    probe.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        transformed = probe(representations)
        predicted_distances = probe.compute_distance_matrix(transformed)

        # Apply mask and compute loss
        valid_mask = masks.unsqueeze(-1) * masks.unsqueeze(-2)  # (batch, num_objects, num_objects)
        masked_pred = predicted_distances * valid_mask
        masked_target = target_distances * valid_mask

        # MSE loss only on valid positions
        loss = torch.sum((masked_pred - masked_target) ** 2) / torch.sum(valid_mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return losses


def evaluate_probe(
    probe: Union[VisualStructuralProbe, SceneGraphComplexityProbe],
    representations: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    probe_type: str,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate the trained probe on test data.

    Args:
        probe: The trained probe
        representations (torch.Tensor): Test representations
        targets (torch.Tensor): Target values (distances or depths)
        masks (torch.Tensor): Masks for valid positions
        probe_type (str): Type of probe ("distance" or "depth")
        device (str): Device to use for evaluation

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    probe.to(device)
    probe.eval()

    representations = representations.to(device)
    targets = targets.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        if probe_type == "distance":
            transformed = probe(representations)
            predictions = probe.compute_distance_matrix(transformed)

            # Flatten and mask for evaluation
            valid_mask = masks.unsqueeze(-1) * masks.unsqueeze(-2)
            pred_flat = predictions[valid_mask.bool()].cpu().numpy()
            target_flat = targets[valid_mask.bool()].cpu().numpy()

        elif probe_type == "depth":
            predictions = probe(representations).squeeze(-1)

            # Flatten and mask for evaluation
            pred_flat = predictions[masks.bool()].cpu().numpy()
            target_flat = targets[masks.bool()].cpu().numpy()

        else:
            raise ValueError("probe_type must be 'distance' or 'depth'")

    # Compute evaluation metrics
    mse = mean_squared_error(target_flat, pred_flat)
    mae = mean_absolute_error(target_flat, pred_flat)
    pearson_corr, _ = pearsonr(target_flat, pred_flat)
    spearman_corr, _ = spearmanr(target_flat, pred_flat)

    return {
        "mse": mse,
        "mae": mae,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
    }


def extract_patch_embeddings(
    model_id: str,
    image_paths: List[str],
    patch_indices_list: List[List[int]],
    device: str = "cuda",
    extract_all_layers: bool = True,
    cache_dir: Optional[str] = None,
) -> Union[List[torch.Tensor], Dict[int, List[torch.Tensor]]]:
    """
    Extract CLIP patch embeddings for objects based on their patch indices from image file paths.

    Args:
        model_id (str): CLIP model identifier
        image_paths (List[str]): List of paths to image files
        patch_indices_list (List[List[int]]): List of patch indices for each image
        device (str): Device to use for inference
        extract_all_layers (bool): If True, extract from all layers. If False, only final layer.
        cache_dir (Optional[str]): Directory to cache model files

    Returns:
        If extract_all_layers=False: List[torch.Tensor], each of shape (num_objects, embedding_dim)
        If extract_all_layers=True: Dict[int, List[torch.Tensor]], keyed by layer index
    """
    # Load CLIP vision model
    vision_model = CLIPVisionModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    ).to(device)
    vision_model.eval()

    # Load processor
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    if extract_all_layers:
        all_layer_embeddings: dict = {}
        # Get number of layers
        if hasattr(vision_model.vision_model.encoder, "layers"):
            num_layers = len(vision_model.vision_model.encoder.layers)
        else:
            # Fallback for older transformers versions
            num_layers = vision_model.config.num_hidden_layers

        # Initialize storage for each layer
        for layer_idx in range(num_layers):
            all_layer_embeddings[layer_idx] = []
    else:
        all_patch_embeddings = []

    with torch.no_grad():
        for image_path, patch_indices in tqdm(zip(image_paths, patch_indices_list), desc="Extracting patch embeddings"):
            # Load and preprocess image
            image = PIL.Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Forward pass through vision model with all hidden states
            outputs = vision_model(**inputs, output_hidden_states=True)

            if extract_all_layers:
                # Extract from all layers
                hidden_states = outputs.hidden_states[
                    1:
                ]  # Skip the first output (embedding projection)  # Tuple of (layer_0, layer_1, ..., layer_n)

                for layer_idx, layer_hidden_state in enumerate(hidden_states):
                    # Shape: (1, num_patches + 1, hidden_size) - +1 for CLS token
                    patch_embeddings = layer_hidden_state.squeeze(0)  # (num_patches + 1, hidden_size)

                    # Extract embeddings for the specific patch indices
                    object_embeddings = []
                    for patch_idx in patch_indices:
                        if 0 <= patch_idx < patch_embeddings.shape[0]:
                            object_embeddings.append(patch_embeddings[patch_idx])
                        else:
                            # Use zero embedding for invalid indices
                            object_embeddings.append(torch.zeros_like(patch_embeddings[0]))

                    # Stack into tensor
                    if object_embeddings:
                        object_embeddings_tensor = torch.stack(object_embeddings)  # (num_objects, hidden_size)
                        all_layer_embeddings[layer_idx].append(object_embeddings_tensor)
                    else:
                        # Handle empty case
                        all_layer_embeddings[layer_idx].append(torch.zeros((0, patch_embeddings.shape[-1])))
            else:
                # Extract only from final layer (existing behavior)
                patch_embeddings = outputs.last_hidden_state.squeeze(0)  # (num_patches + 1, hidden_size)

                # Extract embeddings for the specific patch indices
                object_embeddings = []
                for patch_idx in patch_indices:
                    if 0 <= patch_idx < patch_embeddings.shape[0]:
                        object_embeddings.append(patch_embeddings[patch_idx])
                    else:
                        # Use zero embedding for invalid indices
                        object_embeddings.append(torch.zeros_like(patch_embeddings[0]))

                # Stack into tensor
                if object_embeddings:
                    object_embeddings_tensor = torch.stack(object_embeddings)  # (num_objects, hidden_size)
                    all_patch_embeddings.append(object_embeddings_tensor)
                else:
                    # Handle empty case
                    all_patch_embeddings.append(torch.zeros((0, patch_embeddings.shape[-1])))

    return all_layer_embeddings if extract_all_layers else all_patch_embeddings


def extract_patch_embeddings_from_images(
    model_id: str,
    images: List[PIL.Image.Image],
    patch_indices_list: List[List[int]],
    device: str = "cuda",
    extract_all_layers: bool = True,
    cache_dir: Optional[str] = None,
) -> Union[List[torch.Tensor], Dict[int, List[torch.Tensor]]]:
    """
    Extract CLIP patch embeddings for objects based on their patch indices from PIL Images.

    Args:
        model_id (str): CLIP model identifier
        images (List[PIL.Image.Image]): List of PIL Images from HuggingFace dataset
        patch_indices_list (List[List[int]]): List of patch indices for each image
        device (str): Device to use for inference
        extract_all_layers (bool): If True, extract from all layers. If False, only final layer.
        cache_dir (Optional[str]): Directory to cache model files

    Returns:
        If extract_all_layers=False: List[torch.Tensor], each of shape (num_objects, embedding_dim)
        If extract_all_layers=True: Dict[int, List[torch.Tensor]], keyed by layer index
    """
    # Load CLIP vision model
    vision_model = CLIPVisionModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    ).to(device)
    vision_model.eval()

    # Load processor
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    if extract_all_layers:
        all_layer_embeddings: dict = {}
        # Get number of layers
        if hasattr(vision_model.vision_model.encoder, "layers"):
            num_layers = len(vision_model.vision_model.encoder.layers)
        else:
            # Fallback for older transformers versions
            num_layers = vision_model.config.num_hidden_layers

        # Initialize storage for each layer
        for layer_idx in range(num_layers):
            all_layer_embeddings[layer_idx] = []
    else:
        all_patch_embeddings = []

    with torch.no_grad():
        for image, patch_indices in tqdm(zip(images, patch_indices_list), desc="Extracting patch embeddings"):
            # Preprocess image
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Forward pass through vision model with all hidden states
            outputs = vision_model(**inputs, output_hidden_states=True)

            if extract_all_layers:
                # Extract from all layers
                hidden_states = outputs.hidden_states  # Tuple of (layer_0, layer_1, ..., layer_n)

                for layer_idx, layer_hidden_state in enumerate(hidden_states):
                    # Shape: (1, num_patches + 1, hidden_size) - +1 for CLS token
                    patch_embeddings = layer_hidden_state.squeeze(0)  # (num_patches + 1, hidden_size)

                    # Extract embeddings for the specific patch indices
                    object_embeddings = []
                    for patch_idx in patch_indices:
                        if 0 <= patch_idx < patch_embeddings.shape[0]:
                            object_embeddings.append(patch_embeddings[patch_idx])
                        else:
                            # Use zero embedding for invalid indices
                            object_embeddings.append(torch.zeros_like(patch_embeddings[0]))

                    # Stack into tensor
                    if object_embeddings:
                        object_embeddings_tensor = torch.stack(object_embeddings)  # (num_objects, hidden_size)
                        all_layer_embeddings[layer_idx].append(object_embeddings_tensor)
                    else:
                        # Handle empty case
                        all_layer_embeddings[layer_idx].append(torch.zeros((0, patch_embeddings.shape[-1])))
            else:
                # Extract only from final layer (existing behavior)
                patch_embeddings = outputs.last_hidden_state.squeeze(0)  # (num_patches + 1, hidden_size)

                # Extract embeddings for the specific patch indices
                object_embeddings = []
                for patch_idx in patch_indices:
                    if 0 <= patch_idx < patch_embeddings.shape[0]:
                        object_embeddings.append(patch_embeddings[patch_idx])
                    else:
                        # Use zero embedding for invalid indices
                        object_embeddings.append(torch.zeros_like(patch_embeddings[0]))

                # Stack into tensor
                if object_embeddings:
                    object_embeddings_tensor = torch.stack(object_embeddings)  # (num_objects, hidden_size)
                    all_patch_embeddings.append(object_embeddings_tensor)
                else:
                    # Handle empty case
                    all_patch_embeddings.append(torch.zeros((0, patch_embeddings.shape[-1])))

    return all_layer_embeddings if extract_all_layers else all_patch_embeddings


def pad_and_batch_data(
    patch_embeddings_list: List[torch.Tensor],
    distance_matrices_list: List[torch.Tensor],
    max_objects: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length sequences to create batches.

    Args:
        patch_embeddings_list: List of patch embeddings tensors
        distance_matrices_list: List of distance matrices
        max_objects: Maximum sequence length. If None, uses max in batch.

    Returns:
        Tuple of (padded_embeddings, padded_distances, masks)
    """
    if max_objects is None:
        max_objects = max(emb.shape[0] for emb in patch_embeddings_list)

    batch_size = len(patch_embeddings_list)
    embedding_dim = patch_embeddings_list[0].shape[-1] if batch_size > 0 else 768

    # Initialize padded tensors
    padded_embeddings = torch.zeros(batch_size, max_objects, embedding_dim)
    padded_distances = torch.zeros(batch_size, max_objects, max_objects)
    masks = torch.zeros(batch_size, max_objects)

    for i, (embeddings, distances) in enumerate(zip(patch_embeddings_list, distance_matrices_list)):
        seq_len = embeddings.shape[0]
        if seq_len > 0:
            # Copy actual data
            padded_embeddings[i, :seq_len] = embeddings
            padded_distances[i, :seq_len, :seq_len] = distances
            masks[i, :seq_len] = 1.0

    return padded_embeddings, padded_distances, masks


def prepare_scene_graph_data(
    graph_data: Dict[str, Any], patch_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Prepare scene graph data for structural probing.

    Args:
        graph_data (Dict[str, Any]): Scene graph data with edge_index, num_nodes, and patch indices
        patch_size (int): Patch size to use for extracting patch indices

    Returns:
        Tuple containing:
        - adjacency_matrix (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes)
        - distance_matrix (torch.Tensor): Tree distance matrix
        - patch_indices (List[int]): Patch indices for each object
    """
    if not graph_data or graph_data.get("num_nodes", 0) == 0:
        return torch.zeros((0, 0)), torch.zeros((0, 0)), []

    num_nodes = graph_data["num_nodes"]
    edge_index = graph_data.get("edge_index", [[], []])
    patch_key = f"patch_indices_{patch_size}"
    patch_indices = graph_data.get(patch_key, [])

    # Create adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    if edge_index and len(edge_index) == 2 and len(edge_index[0]) > 0:
        for source, target in zip(edge_index[0], edge_index[1]):
            if 0 <= source < num_nodes and 0 <= target < num_nodes:
                adjacency_matrix[source, target] = 1
                adjacency_matrix[target, source] = 1  # Make undirected

    # Compute tree distances
    distances = compute_tree_distances_from_adjacency(adjacency_matrix)

    return (
        torch.tensor(adjacency_matrix, dtype=torch.float32),
        torch.tensor(distances, dtype=torch.float32),
        patch_indices,
    )


@hydra.main(config_path="../../../configs/model", config_name="structural_probe")
def run_visual_structural_probe(cfg: DictConfig) -> None:
    """
    Run visual structural probing experiments using CLIP patch embeddings.

    This function loads scene graph data with patch indices, extracts CLIP patch embeddings
    for objects, and trains structural probes to predict scene graph distances.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    # Load VG metadata with scene graphs and patch indices
    from datasets import load_dataset

    vg_dataset = load_dataset(
        cfg.data.processed_hf_identifier,
        cache_dir=cfg.data.dataset_cache_dir,
        split=cfg.data.split,
    )

    # Determine image loading method
    use_hf_dataset = hasattr(cfg.data, "vg_image_dataset_hf_identifier") and cfg.data.vg_image_dataset_hf_identifier
    image_id_to_image = {}

    if use_hf_dataset:
        # Load Visual Genome images dataset from HuggingFace
        logger.info("Loading Visual Genome images dataset from HuggingFace...")
        vg_images_dataset = load_dataset(
            cfg.data.vg_image_dataset_hf_identifier,
            cfg.data.vg_image_dataset_hf_config,
            cache_dir=cfg.data.dataset_cache_dir,
            split="train",
        )

        # Create a mapping from image_id to image for faster lookup
        logger.info("Creating image ID to image mapping...")
        for item in tqdm(vg_images_dataset, desc="Building image mapping"):
            image_id_to_image[item["image_id"]] = item["image"]
    else:
        # Use local image directory
        if not hasattr(cfg.data, "vg_image_dir") or not cfg.data.vg_image_dir:
            raise ValueError("Either vg_image_dataset_hf_identifier or vg_image_dir must be specified in config")
        logger.info(f"Using local image directory: {cfg.data.vg_image_dir}")

    # Check if the output directory exists
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results dataframe
    results = pd.DataFrame(
        columns=[
            "model_id",
            "layer",
            "probe_type",
            "graph_type",
            "fold",
            "mse",
            "mae",
            "pearson_correlation",
            "spearman_correlation",
            "probe_rank",
            "num_samples",
        ]
    )

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    logger.info(f"Using device: {device}")

    # Process each model
    for model_id, patch_size in zip(cfg.model_ids, cfg.patch_sizes):
        logger.info(f"Processing model: {model_id}")

        # Get model configuration to determine number of layers
        from transformers import CLIPVisionConfig

        try:
            config = CLIPVisionConfig.from_pretrained(model_id)
            num_layers = config.num_hidden_layers
        except:  # noqa
            # Fallback for models without explicit config
            num_layers = 12  # Default for most CLIP models

        logger.info(f"Model has {num_layers} layers")

        # Process different graph types
        graph_types = ["image_graphs", "action_image_graphs", "spatial_image_graphs"]
        available_graph_types = [gt for gt in graph_types if gt in vg_dataset.column_names]

        if not available_graph_types:
            logger.warning("No scene graph columns found in dataset")
            continue

        for graph_type in available_graph_types:
            logger.info(f"Processing graph type: {graph_type}")

            # Filter dataset to only include samples with valid graphs and collect all data first
            valid_samples = []
            valid_graph_structures = []
            valid_images = []
            valid_patch_indices = []

            # TODO remove
            # Take a subset of vg_dataset for faster processing
            vg_dataset = vg_dataset.select([i for i in list(range(100))])

            for i, sample in enumerate(tqdm(vg_dataset, desc=f"Filtering {graph_type} samples")):
                graph_structure = sample.get(graph_type)
                if not graph_structure or graph_structure.get("num_nodes", 0) == 0:
                    continue

                # Prepare scene graph data
                try:
                    adjacency_matrix, distance_matrix, _ = prepare_scene_graph_data(
                        graph_structure, patch_size=patch_size
                    )
                    if adjacency_matrix.shape[0] == 0:
                        continue

                    # Get image using the appropriate method
                    image_id = sample.get("image_id")

                    if use_hf_dataset:
                        # Get image from the Visual Genome images dataset
                        if image_id not in image_id_to_image:
                            logger.warning(f"Image ID {image_id} not found in VG images dataset")
                            continue
                        image = image_id_to_image[image_id]
                        valid_images.append(image)
                    else:
                        # Construct path to image file
                        image_path = Path(cfg.data.vg_image_dir) / f"{image_id}.jpg"
                        if not image_path.exists():
                            logger.warning(f"Image file not found: {image_path}")
                            continue
                        valid_images.append(str(image_path))

                    valid_samples.append(sample)
                    valid_graph_structures.append((adjacency_matrix, distance_matrix))
                    valid_patch_indices.append(sample.get(graph_type).get(f"patch_indices_{patch_size}", []))

                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue

            logger.info(f"Found {len(valid_samples)} valid samples for {graph_type}")

            # Extract patch embeddings for all layers at once
            logger.info("Extracting patch embeddings from all layers...")

            if use_hf_dataset:
                # Use the function for PIL Images from HuggingFace dataset
                all_layer_embeddings = extract_patch_embeddings_from_images(
                    model_id=model_id,
                    images=valid_images,
                    patch_indices_list=valid_patch_indices,
                    device=device,
                    extract_all_layers=True,
                    cache_dir=cfg.data.model_cache_dir,
                )
            else:
                # Use the function for file paths
                all_layer_embeddings = extract_patch_embeddings(
                    model_id=model_id,
                    image_paths=valid_images,
                    patch_indices_list=valid_patch_indices,
                    device=device,
                    extract_all_layers=True,
                    cache_dir=cfg.data.model_cache_dir,
                )

            # Process each layer
            for layer_idx in tqdm(range(num_layers), desc=f"Processing layers for {model_id}"):
                # Get embeddings for this layer
                layer_embeddings = all_layer_embeddings[layer_idx]
                distance_matrices_list = [structure[1] for structure in valid_graph_structures]

                # Cross-validation
                kf = KFold(n_splits=cfg.cv, shuffle=True, random_state=cfg.random_state)

                for fold, (train_idx, test_idx) in enumerate(kf.split(valid_samples)):

                    # Prepare training data
                    train_embeddings = [layer_embeddings[i] for i in train_idx]
                    train_distances = [distance_matrices_list[i] for i in train_idx]
                    test_embeddings = [layer_embeddings[i] for i in test_idx]
                    test_distances = [distance_matrices_list[i] for i in test_idx]

                    # Pad and batch data
                    X_train, dist_train, mask_train = pad_and_batch_data(train_embeddings, train_distances)
                    X_test, dist_test, mask_test = pad_and_batch_data(test_embeddings, test_distances)

                    embedding_dim = X_train.shape[2]

                    # Train structural probe
                    if "structural" in cfg.probe_types:
                        structural_probe = VisualStructuralProbe(
                            representation_dim=embedding_dim,
                            probe_rank=(cfg.model.probe_rank if cfg.model.probe_rank > 0 else embedding_dim),
                        )

                        # Train the probe
                        train_structural_probe(
                            probe=structural_probe,
                            representations=X_train,
                            target_distances=dist_train,
                            masks=mask_train,
                            learning_rate=cfg.training.learning_rate,
                            num_epochs=cfg.training.num_epochs,
                            device=device,
                        )

                        # Evaluate the probe
                        eval_metrics = evaluate_probe(
                            probe=structural_probe,
                            representations=X_test,
                            targets=dist_test,
                            masks=mask_test,
                            probe_type="distance",
                            device=device,
                        )

                        # Add results
                        result_row = {
                            "model_id": model_id,
                            "layer": layer_idx,
                            "probe_type": "structural",
                            "graph_type": graph_type,
                            "fold": fold,
                            "probe_rank": (cfg.model.probe_rank if cfg.model.probe_rank > 0 else embedding_dim),
                            "num_samples": len(train_idx),
                            **eval_metrics,
                        }
                        results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)

                    # Train complexity probe (predicting graph properties from patch embeddings)
                    if "complexity" in cfg.probe_types:
                        # Extract graph properties for complexity prediction
                        graph_properties_train = []
                        graph_properties_test = []

                        for idx in train_idx:
                            sample = valid_samples[idx]
                            props = extract_scene_graph_properties(sample[graph_type])
                            graph_properties_train.append(list(props.values()))

                        for idx in test_idx:
                            sample = valid_samples[idx]
                            props = extract_scene_graph_properties(sample[graph_type])
                            graph_properties_test.append(list(props.values()))

                        props_train_tensor = torch.tensor(graph_properties_train, dtype=torch.float32)
                        props_test_tensor = torch.tensor(graph_properties_test, dtype=torch.float32)

                        # Use CLS token representations for image-level complexity prediction
                        # We need to extract CLS tokens from the layer embeddings
                        cls_embeddings_train = []
                        cls_embeddings_test = []

                        # Extract CLS tokens (index 0) for training samples
                        for idx in train_idx:
                            # Get the full layer output for this image (includes CLS token)
                            full_layer_output = all_layer_embeddings[layer_idx][idx]
                            # CLS token is at index 0
                            cls_token = full_layer_output[0]  # Shape: (hidden_size,)
                            cls_embeddings_train.append(cls_token)

                        # Extract CLS tokens for test samples
                        for idx in test_idx:
                            full_layer_output = all_layer_embeddings[layer_idx][idx]
                            cls_token = full_layer_output[0]
                            cls_embeddings_test.append(cls_token)

                        X_train_img = torch.stack(cls_embeddings_train)  # (batch_size, hidden_size)
                        X_test_img = torch.stack(cls_embeddings_test)  # (batch_size, hidden_size)

                        complexity_probe = SceneGraphComplexityProbe(
                            representation_dim=embedding_dim,
                            num_properties=props_train_tensor.shape[1],
                        )

                        # Train complexity probe
                        complexity_probe.to(device)
                        X_train_img = X_train_img.to(device)
                        props_train_tensor = props_train_tensor.to(device)

                        optimizer = optim.Adam(complexity_probe.parameters(), lr=cfg.training.learning_rate)

                        complexity_probe.train()
                        for _ in range(cfg.training.num_epochs):
                            optimizer.zero_grad()

                            predictions = complexity_probe(X_train_img)
                            loss = nn.MSELoss()(predictions, props_train_tensor)

                            loss.backward()
                            optimizer.step()

                        # Evaluate complexity probe
                        complexity_probe.eval()
                        X_test_img = X_test_img.to(device)
                        props_test_tensor = props_test_tensor.to(device)

                        with torch.no_grad():
                            test_predictions = complexity_probe(X_test_img)

                        pred_np = test_predictions.cpu().numpy()
                        target_np = props_test_tensor.cpu().numpy()

                        # Compute metrics (averaged across all properties)
                        mse = mean_squared_error(target_np.flatten(), pred_np.flatten())
                        mae = mean_absolute_error(target_np.flatten(), pred_np.flatten())
                        pearson_corr, _ = pearsonr(target_np.flatten(), pred_np.flatten())
                        spearman_corr, _ = spearmanr(target_np.flatten(), pred_np.flatten())

                        # Add results
                        result_row = {
                            "model_id": model_id,
                            "layer": layer_idx,
                            "probe_type": "complexity",
                            "graph_type": graph_type,
                            "fold": fold,
                            "probe_rank": embedding_dim,
                            "num_samples": len(train_idx),
                            "mse": mse,
                            "mae": mae,
                            "pearson_correlation": pearson_corr,
                            "spearman_correlation": spearman_corr,
                        }
                        results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)

    # Save results
    results_file = output_dir / "visual_structural_probe_results.csv"
    results.to_csv(results_file, index=False)
    logger.info(f"Visual structural probing complete. Results saved to {results_file}")

    # Generate layer-wise summary
    if len(results) > 0:
        summary_stats = (
            results.groupby(["model_id", "layer", "probe_type", "graph_type"])
            .agg(
                {
                    "mse": ["mean", "std"],
                    "mae": ["mean", "std"],
                    "pearson_correlation": ["mean", "std"],
                    "spearman_correlation": ["mean", "std"],
                    "num_samples": "first",
                }
            )
            .round(4)
        )

        summary_file = output_dir / "visual_structural_probe_layer_summary.csv"
        summary_stats.to_csv(summary_file)
        logger.info(f"Layer-wise summary saved to {summary_file}")

        # Print best performing layers for each probe type and graph type
        for probe_type in results["probe_type"].unique():
            for graph_type in results["graph_type"].unique():
                subset = results[(results["probe_type"] == probe_type) & (results["graph_type"] == graph_type)]
                if len(subset) > 0:
                    # Find layer with best average Spearman correlation
                    layer_performance = subset.groupby("layer")["spearman_correlation"].mean()
                    best_layer = layer_performance.idxmax()
                    best_score = layer_performance.max()
                    logger.info(
                        f"Best layer for {probe_type} on {graph_type}: "
                        f"Layer {best_layer} (Spearman: {best_score:.4f})"
                    )


if __name__ == "__main__":
    run_visual_structural_probe()

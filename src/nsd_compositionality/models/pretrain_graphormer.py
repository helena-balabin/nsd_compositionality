"""Pretrain Graphormer component on graph data only."""

import logging
import os

import dotenv
import hydra
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import GraphormerConfig, GraphormerModel, GraphormerPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from nsd_compositionality.data.preprocess_graphormer import GraphCLIPDataCollator

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


def build_edge_index(input_edges: torch.Tensor) -> torch.Tensor:
    """
    Build edge_index from Graphormer-style dense edge features.

    :param input_edges: Tensor of shape (batch_size, num_nodes, num_nodes, num_edge_features, 1)
        representing edge features between nodes.
    :return: edge_index tensor of shape (2, num_edges) suitable for edge prediction
        (source nodes, target nodes).
    """
    # [B, N, N, F, 1] -> [B, N, N, F]
    edge_matrix = input_edges.squeeze(-1)

    # Collapse edge feature dim: if any feature > 0, mark edge as existing
    edge_matrix = edge_matrix.sum(dim=-1) > 0  # [B, N, N], boolean

    batch_size, num_nodes, _ = edge_matrix.shape
    device = input_edges.device
    edge_index_list = []

    for b in range(batch_size):
        src, dst = edge_matrix[b].nonzero(as_tuple=True)
        # Offset nodes by batch so embeddings can be flattened
        edges = torch.stack([src + b * num_nodes, dst + b * num_nodes], dim=0)
        edge_index_list.append(edges)

    if len(edge_index_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.cat(edge_index_list, dim=1).to(device)


class GraphormerForEdgePrediction(GraphormerPreTrainedModel):
    def __init__(self, config, num_neg_samples_ratio: float = 0.5):
        """
        Args:
            num_neg_samples_ratio: how many negatives to sample per positive edge
        """
        super().__init__(config)
        self.num_labels = 2
        self.graphormer = GraphormerModel(config)
        self.edge_scorer = nn.Bilinear(config.hidden_size, config.hidden_size, 1)
        self.num_neg_samples_ratio = num_neg_samples_ratio
        self.post_init()

    def _sample_negative_edges(self, num_nodes, edge_index, num_samples):
        """Vectorized negative sampling - much faster."""
        device = edge_index.device

        # Create adjacency matrix for fast lookup
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
        adj[edge_index[0], edge_index[1]] = True
        adj.fill_diagonal_(True)  # Exclude self-loops

        # Find all possible negative edges at once
        all_pairs = torch.cartesian_prod(torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device))

        # Filter out existing edges and self-loops
        valid_negatives = ~adj[all_pairs[:, 0], all_pairs[:, 1]]
        negative_candidates = all_pairs[valid_negatives]

        # Sample from candidates
        if len(negative_candidates) < num_samples:
            neg_edges = negative_candidates
        else:
            indices = torch.randperm(len(negative_candidates), device=device)[:num_samples]
            neg_edges = negative_candidates[indices]

        return neg_edges.t()  # Return as (2, num_samples)

    def _sample_negatives_vectorized(self, batch_size, num_nodes, pos_edges, num_neg, device):
        """Much faster vectorized negative sampling."""
        # Create a set of existing edges for fast lookup
        edge_set = set(zip(pos_edges[0].cpu().tolist(), pos_edges[1].cpu().tolist()))

        # Generate random pairs efficiently
        neg_edges = []
        max_attempts = num_neg * 10  # Prevent infinite loops
        attempts = 0

        while len(neg_edges) < num_neg and attempts < max_attempts:
            # Generate batch of candidates
            batch_candidates = min(num_neg * 2, num_neg - len(neg_edges))
            src_candidates = torch.randint(0, batch_size * num_nodes, (batch_candidates,), device=device)
            dst_candidates = torch.randint(0, batch_size * num_nodes, (batch_candidates,), device=device)

            # Filter valid candidates
            for i in range(batch_candidates):
                src, dst = src_candidates[i].item(), dst_candidates[i].item()
                if src != dst and (src, dst) not in edge_set:
                    neg_edges.append([src, dst])
                    if len(neg_edges) >= num_neg:
                        break

            attempts += batch_candidates

        if len(neg_edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        return torch.tensor(neg_edges[:num_neg], device=device).t()

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        perturb=None,
        masked_tokens=None,
        return_dict=True,
    ):
        """
        Arguments mirror GraphormerModel, but based on edge prediction.

        :param input_nodes: Node feature tensor of shape (batch_size, num_nodes)
        :param input_edges: Edge index tensor of shape (batch_size, num_edges,
            2) where each edge is represented as (source_node, target_node)
        :param attn_bias: Attention bias tensor of shape (batch_size, num_nodes,
        :param in_degree: In-degree tensor of shape (batch_size, num_nodes)
        :param out_degree: Out-degree tensor of shape (batch_size, num_nodes)
        :param spatial_pos: Spatial position tensor of shape (batch_size, num_nodes,
            num_nodes)
        :param attn_edge_type: Attention edge type tensor of shape (batch_size,
            num_nodes, num_nodes)
        :param perturb: Optional perturbation tensor for adversarial training
        :param masked_tokens: Optional tensor indicating masked nodes for
            masked node prediction
        :param return_dict: Whether to return a SequenceClassifierOutput or a
            plain tuple
        :return: SequenceClassifierOutput with loss and logits for edge
            predictions
        """
        # Encode nodes
        outputs = self.graphormer(
            input_nodes=input_nodes,
            input_edges=input_edges,
            attn_bias=attn_bias,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos,
            attn_edge_type=attn_edge_type,
            perturb=perturb,
            masked_tokens=masked_tokens,
            return_dict=return_dict,
        )
        node_reps = outputs.last_hidden_state

        batch_size, num_nodes, hidden_dim = node_reps.shape

        # Build positive edges more efficiently
        edge_index = build_edge_index(input_edges)

        if edge_index.size(1) == 0:
            # No edges found, return dummy loss
            dummy_logits = torch.zeros(1, device=node_reps.device)
            loss = F.binary_cross_entropy_with_logits(dummy_logits, torch.zeros_like(dummy_logits))

            if not return_dict:
                return (loss, dummy_logits)
            return SequenceClassifierOutput(loss=loss, logits=dummy_logits)

        # Vectorized negative sampling
        num_pos_edges = edge_index.size(1)
        num_neg_edges = int(num_pos_edges * self.num_neg_samples_ratio)

        # Fast negative sampling - sample from all possible pairs then filter
        neg_edges = self._sample_negatives_vectorized(
            batch_size, num_nodes, edge_index, num_neg_edges, node_reps.device
        )

        # Flatten representations once
        node_reps_flat = node_reps.view(-1, hidden_dim)

        # Compute all scores at once
        all_edges = torch.cat([edge_index, neg_edges], dim=1) if neg_edges.size(1) > 0 else edge_index
        src_nodes, dst_nodes = all_edges

        all_scores = self.edge_scorer(node_reps_flat[src_nodes], node_reps_flat[dst_nodes]).squeeze(-1)

        # Split scores and create labels
        pos_scores = all_scores[:num_pos_edges]
        neg_scores = all_scores[num_pos_edges:] if neg_edges.size(1) > 0 else torch.empty(0, device=all_scores.device)

        # Combine logits and labels
        logits = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            return (loss, logits)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


def pretrain_single_graph_type(cfg: DictConfig, graph_type: str):
    """Pretrain Graphormer on a single graph type."""

    logger.info(f"Starting pretraining for graph type: {graph_type}")

    # Setup MLflow run for this graph type
    run_name = f"graphormer_pretrain_{graph_type}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "graph_type": graph_type,
                "dataset": cfg.data.hf_dataset_identifier,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.learning_rate,
                "epochs": cfg.training.epochs,
                "hidden_size": cfg.model.hidden_size,
                "num_hidden_layers": cfg.model.num_hidden_layers,
            }
        )

        # Load dataset
        logger.info(f"Loading dataset: {cfg.data.hf_dataset_identifier}")
        dataset = load_dataset(
            cfg.data.hf_dataset_identifier,
            cache_dir=cfg.data.cache_dir,
            split=cfg.data.split,
        )

        # Filter for samples without text and with valid graphs for this graph type
        logger.info(f"Filtering dataset for {graph_type} samples...")
        original_size = len(dataset)
        dataset = dataset.filter(
            lambda x: (x.get("sentences_raw") is None or x["sentences_raw"] == "")
            and x[graph_type]["num_nodes"] > 0
            and len(x[graph_type]["edge_index"][0]) > 0
        )

        logger.info(f"Filtered from {original_size} to {len(dataset)} samples for {graph_type} pretraining")

        if len(dataset) == 0:
            raise ValueError(f"No valid {graph_type} samples found after filtering!")

        # Prepare dataset - only keep graph inputs for this graph type
        def prepare_graph_only(example):
            return {"graph_input": example[graph_type]}

        dataset = dataset.map(
            prepare_graph_only,
            num_proc=cfg.data.num_proc,
            remove_columns=[col for col in dataset.column_names if col != "graph_input"],
            desc=f"Preparing {graph_type} graphs",
        )

        # Initialize Graphormer model directly
        graphormer_config = GraphormerConfig(
            hidden_size=cfg.model.hidden_size,
            embedding_dim=cfg.model.embedding_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            dropout=cfg.model.dropout,
        )

        logger.info(f"Initializing Graphormer with config: {graphormer_config}")
        model = GraphormerForEdgePrediction(graphormer_config)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory for this graph type
        output_dir = os.path.join(cfg.output_dir, f"pretrained_graphormer_{graph_type}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            warmup_ratio=cfg.training.warmup_ratio,
            num_train_epochs=cfg.training.epochs,
            logging_steps=cfg.training.logging_steps,
            save_strategy="epoch",
            eval_strategy="no",
            push_to_hub=cfg.training.push_to_hub,
            hub_model_id=f"{cfg.training.hub_model_id_base}_{graph_type}" if cfg.training.push_to_hub else None,
            report_to="mlflow",
            dataloader_drop_last=True,
            remove_unused_columns=False,
        )

        # Use the existing GraphCLIPDataCollator with on_the_fly_processing=True
        # since the graphs are not preprocessed yet
        data_collator = GraphCLIPDataCollator(
            spatial_pos_max=20,
            on_the_fly_processing=True,  # Process graphs on the fly
            unwrap_dict=True,
        )

        # Train with simple masked node prediction objective (built into Graphormer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        logger.info(f"Starting training for {graph_type}...")
        # Train
        train_result = trainer.train()

        # Log final metrics
        mlflow.log_metrics(
            {
                "final_loss": train_result.training_loss,
                "train_samples": len(dataset),
            }
        )

        # Save model to HuggingFace Hub if configured
        if cfg.training.push_to_hub:
            hub_model_id = f"{cfg.training.hub_model_id_base}_{graph_type}"
            logger.info(f"Pushing {graph_type} model to HuggingFace Hub: {hub_model_id}")
            trainer.push_to_hub()

            # Log the hub model ID to MLflow
            mlflow.log_param("hub_model_id", hub_model_id)

        logger.info(f"Training completed for {graph_type}!")
        return model


@hydra.main(config_path="../../../configs/model", config_name="pretrain_graphormer", version_base="1.1")
def pretrain_graphormer(cfg: DictConfig):
    """Pretrain Graphormer on graph data only for all specified graph types."""

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Get graph types to train (default to all three)
    graph_types = getattr(cfg.training, "graph_types", ["image_graphs", "action_image_graphs", "spatial_image_graphs"])

    logger.info(f"Will pretrain Graphormer for graph types: {graph_types}")

    models = {}
    for graph_type in graph_types:
        model = pretrain_single_graph_type(cfg, graph_type)
        models[graph_type] = model

    logger.info(f"Completed pretraining for {len(models)} graph types: {list(models.keys())}")
    return models


if __name__ == "__main__":
    pretrain_graphormer()

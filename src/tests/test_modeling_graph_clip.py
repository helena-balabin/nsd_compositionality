"""Tests for the GraphCLIPModel class."""

import torch
from datasets import load_dataset
from transformers import GraphormerConfig

from nsd_compositionality.data.preprocess_graphormer import GraphCLIPDataCollator, preprocess_item
from nsd_compositionality.models.graph_clip_model.configuration_graph_clip import GraphCLIPConfig
from nsd_compositionality.models.graph_clip_model.modeling_graph_clip import GraphCLIPModel


def test_graph_clip_model_forward():
    # Create a mock configuration
    graphormer_config = GraphormerConfig()
    config = GraphCLIPConfig(
        graph_config=graphormer_config,
        graph_pair_type="text",
    )
    model = GraphCLIPModel(config)

    # Mock inputs
    input_ids = torch.randint(0, 100, (2, 50))  # Batch size 2, sequence length 50
    pixel_values = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 channels, 224x224 image

    # Get mock graph input
    graph_dataset = load_dataset("OGB/ogbg-molhiv", split="train[:2]")
    # Use the GraphCLIPDataCollator to process all the features
    graph_collator = GraphCLIPDataCollator(
        spatial_pos_max=20,
        on_the_fly_processing=False,
    )
    # Put all features together: Graph, text and image
    all_features = [
        {
            "input_ids": input_ids[i],
            "pixel_values": pixel_values[i],
            "graph_input": preprocess_item(graph_dataset[i]),
        }
        for i in range(2)
    ]
    # Use the collator to process the batch
    batch = graph_collator(all_features)

    # Forward pass
    outputs = model(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        graph_input=batch["graph_input"],
        return_loss=True,
    )

    # Assertions
    assert outputs.loss is not None
    assert outputs.logits_image_text.shape == (2, 2)
    assert outputs.logits_graph_pair.shape == (2, 2)
    assert outputs.image_embeds.shape == (2, 512)
    assert outputs.text_embeds.shape == (2, 512)
    assert outputs.graph_embeds.shape == (2, 512)

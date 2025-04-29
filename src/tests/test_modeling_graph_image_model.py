"""Tests for the GraphCLIPModel class."""

import torch
from datasets import load_dataset
from transformers import CLIPConfig, GraphormerConfig

from nsd_compositionality.data.preprocess_graphormer import GraphormerDataCollator, preprocess_item
from nsd_compositionality.models.modeling_graph_image_model import GraphCLIPModel


def test_graph_clip_model_forward():
    # Create a mock configuration
    graphormer_config = GraphormerConfig()
    config = CLIPConfig(
        graph_config=graphormer_config,
        graph_pair_type="text",
    )
    model = GraphCLIPModel(config)

    # Mock inputs
    input_ids = torch.randint(0, 100, (2, 50))  # Batch size 2, sequence length 50
    pixel_values = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 channels, 224x224 image

    # Get mock graph input
    graph_dataset = load_dataset("OGB/ogbg-molhiv", split="train[:2]")
    graph_dataset = graph_dataset.map(preprocess_item, batched=False)

    # Use GraphormerDataCollator to preprocess the graph data
    data_collator = GraphormerDataCollator()
    graph_inputs = data_collator([graph_dataset[i] for i in range(len(graph_dataset))])

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        graph_input=graph_inputs,
        return_loss=True,
    )

    # Assertions
    assert outputs.loss is not None
    assert outputs.logits_image_text.shape == (2, 2)
    assert outputs.logits_graph_pair.shape == (2, 2)
    assert outputs.image_embeds.shape == (2, 512)
    assert outputs.text_embeds.shape == (2, 512)
    assert outputs.graph_embeds.shape == (2, 512)

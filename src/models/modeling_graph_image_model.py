"""Contrastive Learning-Based Graph, Image, and Text Model."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPModel, GraphormerModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, BaseModelOutputWithPooling, ModelOutput
from transformers.models.clip.modeling_clip import clip_loss


class GraphCLIPOutput(ModelOutput):
    """
    Custom output class for GraphCLIPModel.

    Attributes:
        loss (torch.FloatTensor, optional): Loss value if return_loss is True.
        logits_image_text (torch.FloatTensor): Logits for image-text pairs.
        logits_graph_pair (torch.FloatTensor): Logits for graph-text or graph-image pairs.
        image_embeds (torch.FloatTensor): Image embeddings.
        graph_embeds (torch.FloatTensor): Graph embeddings.
        text_embeds (torch.FloatTensor): Text embeddings.
        vision_model_output (BaseModelOutputWithPooling): Output from the vision model.
        text_model_output (BaseModelOutputWithPooling): Output from the text model.
        graph_model_output (BaseModelOutputWithNoAttention): Output from the graph model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_image_text: torch.FloatTensor
    logits_graph_pair: torch.FloatTensor
    image_embeds: torch.FloatTensor
    graph_embeds: torch.FloatTensor
    text_embeds: torch.FloatTensor
    vision_model_output: BaseModelOutputWithPooling
    text_model_output: BaseModelOutputWithPooling
    graph_model_output: BaseModelOutputWithNoAttention


class GraphCLIPModel(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        graph_config = config.graph_config

        # Initialize Graphormer model
        self.graph_model = GraphormerModel._from_config(graph_config)

        # Projection layer for graph embeddings
        self.graph_projection = nn.Linear(graph_config.hidden_size, self.projection_dim, bias=False)

        # Determine the graph pair type (either "text" or "image")
        self.graph_pair_type = config.graph_pair_type  # Should be "text" or "image"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        graph_input: Optional[dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GraphCLIPOutput]:
        """
        Forward pass of GraphCLIP Model with three modalities: image, graph, and text.

        Args:
            input_ids (torch.LongTensor): Tokenized text input IDs.
            pixel_values (torch.FloatTensor): Batch of images.
            graph_input (dict): Dictionary of inputs for the Graphormer encoder.
            attention_mask (torch.LongTensor, optional): Attention mask for text and graph encoders.
            position_ids (torch.LongTensor, optional): Position IDs for text encoder.
            return_loss (bool, optional): Whether to compute the contrastive loss.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a ModelOutput object.

        Returns:
            GraphCLIPOutput: Custom output object containing logits and embeddings.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Process images through the CLIP vision encoder
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[1]  # Pooled output
        image_embeds = self.visual_projection(image_embeds)

        # Process text input through CLIP text encoder
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_embeds = text_outputs[1]  # Pooled output
        text_embeds = self.text_projection(text_embeds)

        # Process graph input through Graphormer
        graph_outputs = self.graph_model(
            **graph_input,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        graph_embeds = graph_outputs.last_hidden_state.mean(dim=1)  # Mean pooling of the last hidden state
        graph_embeds = self.graph_projection(graph_embeds)

        # Normalize the projected features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        graph_embeds = graph_embeds / graph_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute scaled cosine similarity logits
        logit_scale = self.logit_scale.exp()
        logits_image_text = logit_scale * torch.matmul(image_embeds, text_embeds.t())

        # Compute graph pair logits based on the specified pair type
        if self.graph_pair_type == "text":
            logits_graph_pair = logit_scale * torch.matmul(graph_embeds, text_embeds.t())
        elif self.graph_pair_type == "image":
            logits_graph_pair = logit_scale * torch.matmul(graph_embeds, image_embeds.t())
        else:
            raise ValueError("Invalid graph_pair_type. Must be 'text' or 'image'.")

        loss = None
        if return_loss:
            # Compute contrastive loss for the specified pairs
            loss_image_text = clip_loss(logits_image_text)
            loss_graph_pair = clip_loss(logits_graph_pair)
            loss = (loss_image_text + loss_graph_pair) / 2

        if not return_dict:
            output = (
                logits_image_text,
                logits_graph_pair,
                image_embeds,
                graph_embeds,
                text_embeds,
                vision_outputs,
                text_outputs,
                graph_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return GraphCLIPOutput(
            loss=loss,
            logits_image_text=logits_image_text,
            logits_graph_pair=logits_graph_pair,
            image_embeds=image_embeds,
            graph_embeds=graph_embeds,
            text_embeds=text_embeds,
            vision_model_output=vision_outputs,
            text_model_output=text_outputs,
            graph_model_output=graph_outputs,
        )

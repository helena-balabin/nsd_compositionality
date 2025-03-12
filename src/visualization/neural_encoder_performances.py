"""Visualize the aggregated results of the neural encoder in a connected scatter plot."""

import logging
import os

import hydra
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _layer_index(layer_name: str) -> int:
    """Extract numeric index from layer name; handle empty layer strings."""
    if "_layer_" in layer_name:
        return int(layer_name.split("_")[-1])
    return -1


@hydra.main(config_path="../../configs/visualization", config_name="neural_encoder_performances")
def visualize_neural_encoder_results(cfg: DictConfig) -> None:
    """
    Read the neural encoder results from CSV, average across subjects, and plot:
    1) Pairwise accuracy vs. layer
    2) Pearson correlation vs. layer
    """
    csv_path = os.path.abspath(cfg.data.results_csv)
    output_dir = os.path.abspath(cfg.data.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading results from {csv_path}")
    df = pd.read_csv(csv_path)

    # Average mean performances and standard deviations across subjects
    grouped = df.groupby(["model_id", "layer"], as_index=False).agg(
        {
            "pairwise_accuracy_mean": "mean",
            "pairwise_accuracy_std": "mean",
            "pearson_correlation_mean": "mean",
            "pearson_correlation_std": "mean",
        }
    )

    # Sort by model and layer index
    grouped["layer_idx"] = grouped["layer"].apply(_layer_index)
    grouped.sort_values(by=["model_id", "layer_idx"], inplace=True)

    # Create a separate figure for each metric
    metrics = [
        ("pairwise_accuracy_mean", "pairwise_accuracy_std", "pairwise accuracy", px.colors.sequential.Purples),
        ("pearson_correlation_mean", "pearson_correlation_std", "pearson correlation", px.colors.sequential.Oranges),
    ]

    # Iterate over metrics and create a figure for each
    for mean_col, std_col, metric_name, color_palette in metrics:
        # if there is only one model, use the middle color
        if len(grouped["model_id"].unique()) == 1:
            color_palette = [color_palette[len(color_palette) // 2]]
        fig = go.Figure()
        for i, model_id in enumerate(grouped["model_id"].unique()):
            model_data = grouped[grouped["model_id"] == model_id]
            x = model_data["layer_idx"]
            y = model_data[mean_col]
            yerr = model_data[std_col]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=model_id,
                    showlegend=True,
                    line=dict(color=color_palette[::-1][i % len(color_palette)]),
                    error_y=dict(type="data", array=yerr, visible=True),
                )
            )

        fig.update_layout(
            title=f"{metric_name} across layers",
            title_x=0.5,
            xaxis_title="layer index",
            yaxis_title=metric_name,
            plot_bgcolor="white",
            legend_title="model ID",
            legend=dict(orientation="h", y=1.1),
        )

        output_path = os.path.join(output_dir, f"{metric_name.replace(' ', '_').lower()}.png")
        fig.write_image(output_path)
        logger.info(f"Saved {metric_name} figure to {output_path}")


if __name__ == "__main__":
    visualize_neural_encoder_results()

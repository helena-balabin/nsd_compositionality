"""Visualize the aggregated results of the neural encoder in a connected scatter plot."""

import logging
import os

import hydra
import pandas as pd
import plotly.graph_objects as go
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _layer_index(layer_name: str) -> int:
    """Extract numeric index from layer name; handle empty layer strings."""
    if "_layer_" in layer_name:
        return int(layer_name.split("_")[-1])
    return -1


@hydra.main(config_path="../../../configs/visualization", config_name="neural_encoder_performances")
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

    # Create one plot per model
    unique_models = grouped["model_id"].unique()
    for model_id in unique_models:
        model_data = grouped[grouped["model_id"] == model_id]

        # Create separate figure for Pairwise Accuracy
        fig_pairwise = go.Figure()
        fig_pairwise.add_trace(
            go.Scatter(
                x=model_data["layer_idx"],
                y=model_data["pairwise_accuracy_mean"],
                mode="lines+markers",
                name="Pairwise Accuracy",
                error_y=dict(type="data", array=model_data["pairwise_accuracy_std"], visible=True),
                line={"color": "#589dd6"},
            )
        )
        fig_pairwise.update_layout(
            title=f"Pairwise Accuracy for: {model_id.split('/')[-1]}",
            title_x=0.5,
            plot_bgcolor="white",
            legend_title="Metrics",
            xaxis_title="Layer Index",
            yaxis_title="Pairwise Accuracy",
            yaxis=dict(tickfont=dict(size=18)),
            xaxis=dict(tickfont=dict(size=18)),
        )
        output_path_pairwise = os.path.join(output_dir, f"{model_id.replace('/', '-')}_pairwise.png")
        fig_pairwise.write_image(output_path_pairwise)
        logger.info(f"Saved Pairwise Accuracy figure for model {model_id} to {output_path_pairwise}")

        # Create separate figure for Pearson Correlation
        fig_pearson = go.Figure()
        fig_pearson.add_trace(
            go.Scatter(
                x=model_data["layer_idx"],
                y=model_data["pearson_correlation_mean"],
                mode="lines+markers",
                name="Pearson Correlation",
                error_y=dict(type="data", array=model_data["pearson_correlation_std"], visible=True),
                line={"color": "#6ea2cc"},
            )
        )
        fig_pearson.update_layout(
            title=f"Pearson Correlation for: {model_id.split('/')[-1]}",
            title_x=0.5,
            plot_bgcolor="white",
            legend_title="Metrics",
            xaxis_title="Layer Index",
            yaxis_title="Pearson Correlation",
            yaxis=dict(tickfont=dict(size=18)),
            xaxis=dict(tickfont=dict(size=18)),
        )
        output_path_pearson = os.path.join(output_dir, f"{model_id.replace('/', '-')}_pearson.png")
        fig_pearson.write_image(output_path_pearson)
        logger.info(f"Saved Pearson Correlation figure for model {model_id} to {output_path_pearson}")

    # Create one plot per metric including all models
    metrics = [
        ("pairwise_accuracy_mean", "pairwise_accuracy_std", "Pairwise Accuracy"),
        ("pearson_correlation_mean", "pearson_correlation_std", "Pearson Correlation"),
    ]
    # Get colors accoridng to the number of models from a color palette
    colors = ["#589dd6", "#ae7fb5", "#7cb8e0", "#8fd1e9", "#a3d9f0", "#b7e0f7", "#c9e7fc"][: len(unique_models)]

    for metric_mean, metric_std, metric_name in metrics:
        fig = go.Figure()
        for color, model_id in zip(colors, unique_models):
            model_data = grouped[grouped["model_id"] == model_id]
            fig.add_trace(
                go.Scatter(
                    x=model_data["layer_idx"],
                    y=model_data[metric_mean],
                    mode="lines+markers",
                    name=model_id.split("/")[-1],
                    error_y=dict(type="data", array=model_data[metric_std], visible=True),
                    line={"color": color},
                )
            )
        fig.update_layout(
            title=f"{metric_name} Across All Models",
            title_x=0.5,
            plot_bgcolor="white",
            legend_title="Models",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.75,
                xanchor="center",
                x=0.5,
                title_font=dict(size=18),
                font=dict(size=16),
            ),
            xaxis_title="Layer Index",
            yaxis_title=metric_name,
            yaxis=dict(tickfont=dict(size=18)),
            xaxis=dict(tickfont=dict(size=18)),
            width=800,
            height=400,
        )
        output_path = os.path.join(output_dir, f"all_models_{metric_name.replace(' ', '_').lower()}.png")
        fig.write_image(output_path)
        logger.info(f"Saved {metric_name} figure for all models to {output_path}")


if __name__ == "__main__":
    visualize_neural_encoder_results()

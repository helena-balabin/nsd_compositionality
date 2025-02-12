import logging
import os

import hydra
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

color_palettes = [
    px.colors.sequential.Greens,
    px.colors.sequential.Blues,
    px.colors.sequential.Reds,
    px.colors.sequential.Greys,
    px.colors.sequential.Plotly3,
    px.colors.sequential.Viridis,
    px.colors.sequential.Cividis,
]


def _layer_index(layer_name: str) -> int:
    """Extract numeric index from layer name; handle empty layer strings."""
    if "_layer_" in layer_name:
        return int(layer_name.split("_")[-1])
    return -1


@hydra.main(config_path="../../configs/visualization", config_name="graph_measures_performances")
def visualize_graph_measures(cfg: DictConfig) -> None:
    """
    Read the graph measure probing results from CSV and create one plot per graph feature.
    Only one performance measure is used (defined by cfg.scoring) across multiple graph features.
    """
    csv_path = os.path.abspath(cfg.data.results_csv)
    output_dir = os.path.abspath(cfg.data.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading results from {csv_path}")
    df = pd.read_csv(csv_path)

    # Ensure layer indices are computed for sorting purposes.
    df["layer_idx"] = df["layer"].apply(_layer_index)
    df.sort_values(by=["model_id", "layer_idx"], inplace=True)

    # Identify the measure columns based on cfg.scoring.
    score_mean_col = f"{cfg.scoring}_mean"
    score_std_col = f"{cfg.scoring}_std"

    # Adapt the color palette to the number of models (repeating or slicing if necessary)
    if len(df["model_id"].unique()) > len(color_palettes):
        color_palettes_adapted = color_palettes * (len(df["model_id"].unique()) // len(color_palettes) + 1)
    else:
        color_palettes_adapted = color_palettes

    # Loop over every unique graph feature (target variable) in the results
    # One feature = one plot
    for feature, color_palette in zip(
        df["target_variable"].unique(),
        color_palettes_adapted[: len(df["target_variable"].unique())],
    ):
        model_data = df[df["target_variable"] == feature]

        # Define a color in case there is only one model
        if len(model_data["model_id"].unique()) == 1:
            color_palette = [color_palette[len(color_palette) // 2]]
        fig = go.Figure()

        # Loop over all possible models - one line per model
        for i, model_id in enumerate(model_data["model_id"].unique()):
            model_data = model_data[model_data["model_id"] == model_id]
            x = model_data["layer_idx"]
            y = model_data[score_mean_col]
            yerr = model_data[score_std_col]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=model_id,
                    showlegend=True,
                    line=dict(color=color_palette[i % len(color_palette)]),
                    error_y=dict(type="data", array=yerr, visible=True),
                )
            )

        fig.update_layout(
            title=f"{feature.replace('_', ' ')} across layers",
            title_x=0.5,
            xaxis_title="layer index",
            yaxis_title=feature.replace("_", " "),
            plot_bgcolor="white",
            legend_title="model ID",
            legend=dict(orientation="h", y=1.1),
        )

        output_path = os.path.join(output_dir, f"{feature}.png")
        fig.write_image(output_path)

        logger.info(f"Saved plot for '{feature}' at {output_path}")


if __name__ == "__main__":
    visualize_graph_measures()

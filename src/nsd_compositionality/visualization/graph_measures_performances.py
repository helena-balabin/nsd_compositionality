import logging
import os

import hydra
import pandas as pd
import plotly.graph_objects as go
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

color_palettes = [
    ["#d1913f", "#dbad76", "#e6be8f"],
    ["#f3724d", "#ea947b", "#f3b6a4"],
]


def _layer_index(layer_name: str) -> int:
    """Extract numeric index from layer name; handle empty layer strings."""
    if "_layer_" in layer_name:
        return int(layer_name.split("_")[-1])
    return -1


@hydra.main(config_path="../../../configs/visualization", config_name="graph_measures_performances")
def visualize_graph_measures(cfg: DictConfig) -> None:
    """
    Read the graph measure probing results from CSV and create one plot per model.
    Each plot contains one line per graph feature (target variable).
    The performance measure used is defined by cfg.scoring.
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

    # Loop over every unique model, creating one plot per model.
    for model_idx, model_id in enumerate(df["model_id"].unique()):
        model_data = df[df["model_id"] == model_id]
        fig = go.Figure()

        # Loop over each graph feature (target variable) for the current model.
        for j, feature in enumerate(cfg.target_variables):
            sub_df = model_data[model_data["target_variable"] == feature]
            x = sub_df["layer_idx"]
            y = sub_df[score_mean_col]
            yerr = sub_df[score_std_col]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=feature.replace("sg", "").replace("_filtered", "").replace("_", " "),
                    showlegend=True,
                    line=dict(color=color_palettes[model_idx][j % len(color_palettes[model_idx])]),
                    error_y=dict(type="data", array=yerr, visible=True),
                )
            )

        fig.update_layout(
            title=f"Graph Measures for: {model_id.split('/')[-1]}",
            title_x=0.5,
            xaxis_title="Layer Index",
            yaxis_title=cfg.scoring.replace("_", " ").capitalize(),
            plot_bgcolor="white",
            legend_title="Graph Feature",
            legend=dict(orientation="h", y=1.1),
            font=dict(size=14),  # Increased font size
            yaxis=dict(tickfont=dict(size=18)),
            xaxis=dict(tickfont=dict(size=18)),
        )

        output_path = os.path.join(output_dir, f"{model_id.replace('/', '-')}_graph_measures.png")
        fig.write_image(output_path)
        logger.info(f"Saved plot for model '{model_id}' at {output_path}")


if __name__ == "__main__":
    visualize_graph_measures()

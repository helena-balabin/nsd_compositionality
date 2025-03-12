import logging
import os

import hydra
import pandas as pd
import plotly.express as px
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs/visualization", config_name="mvpa_decoder_performances")
def create_barplot(cfg: DictConfig) -> None:
    """
    Read the MVPA decoder results from CSV (saved by mvpa_decoder) and create a bar plot
    of performances, averaged across subjects for each target variable.
    """
    csv_path = os.path.abspath(cfg.data.results_csv)
    output_dir = os.path.abspath(cfg.data.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading results from {csv_path}")
    df = pd.read_csv(csv_path)

    score_mean_col = f"{cfg.scoring}_mean"
    score_std_col = f"{cfg.scoring}_std"

    # Aggregate performance by target_variable across subjects
    grouped = df.groupby("target_variable").agg({score_mean_col: "mean", score_std_col: "mean"}).reset_index()

    # Create the barplot
    fig = px.bar(
        grouped,
        x="target_variable",
        y=score_mean_col,
        error_y=score_std_col,
        title="MVPA Decoder Performances (Averaged Across Subjects)",
        labels={"target_variable": "Target Variable", score_mean_col: cfg.scoring},
    )
    fig.update_layout(plot_bgcolor="white")

    output_path = os.path.join(output_dir, "mvpa_decoder_barplot.png")
    fig.write_image(output_path)
    logger.info(f"Barplot saved at {output_path}")


if __name__ == "__main__":
    create_barplot()

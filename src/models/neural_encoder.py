"""Neural encoder for NSD data and CLIP-based models, based on precomputed embeddings."""

import logging
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nsd_access import NSDAccess
from omegaconf import DictConfig
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore", module="sklearn")


@hydra.main(config_path="../../configs/model", config_name="neural_encoder")
def run_neural_encoder(cfg: DictConfig) -> None:
    """
    Run a neural encoding analysis on NSD data using a Hugging Face model
    specified by model identifiers in cfg.huggingface.model_ids.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    nsd_dir = Path(cfg.data.large_data_path) / cfg.data.nsd_directory
    nsd = NSDAccess(nsd_dir)

    # Load metadata (e.g., text prompts or other info from NSD)
    nsd_vg_metadata = pd.read_csv(nsd_dir / "nsd_vg" / "nsd_vg_metadata.csv")

    # Check if the output directory exists
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each subject
    for subject in cfg.subjects:
        logger.info(f"Processing subject {subject}")

        all_betas = []
        # Gather data from all sessions
        for session in range(1, cfg.max_sessions + 1):
            session_betas = nsd.read_betas(
                subject,
                session_index=session,
                data_format=cfg.data.data_format,
                data_type=cfg.data.data_type,
            )
            session_betas = zscore(session_betas, axis=-1)
            session_betas = session_betas.astype(np.float16)
            session_betas = np.nan_to_num(session_betas)
            all_betas.append(session_betas)

        # Full data for subject
        betas = np.concatenate(all_betas, axis=-1)

        subject_num = subject.replace("subj0", "subject")
        trial_info = nsd_vg_metadata[nsd_vg_metadata[subject_num] > 0]

        # Unpivot to get trial_index
        trial_info = trial_info.melt(
            id_vars=[c for c in trial_info.columns if "subject" not in c],
            value_vars=[f"{subject_num}_rep0", f"{subject_num}_rep1", f"{subject_num}_rep2"],
            var_name="rep",
            value_name="trial_index",
        )

        trial_info_short = trial_info[trial_info["trial_index"] < betas.shape[-1]]
        trial_info_short = trial_info_short.sort_values(by="trial_index").reset_index(drop=True)

        # Extract betas for valid trials
        Y = betas[:, :, :, trial_info_short["trial_index"].values - 1]
        # Use the nsdgeneral mask
        if cfg.nsdgeneral_mask:
            # Use a much smaller mask in the subject native space
            # Read the atlas results for the given subject
            atlas_results = nsd.read_atlas_results(
                subject,
                data_format=cfg.data.data_format,
                atlas="nsdgeneral",
            )[0]
            # Set all -1 values to 0 and filter the fMRI data
            atlas_results[atlas_results == -1] = 0
            Y = Y[atlas_results == 1].T

        # Iterate over the models that we want to test
        for model_id in cfg.model_ids:
            # Replace the text-based inference with loading precomputed embeddings
            # The embeddings have as many entries as trial_info_short
            embedding_file = Path(cfg.data.embedding_dir) / f"{model_id.replace('/', '_')}_embeddings.npy"
            X_embeddings_initial = np.load(embedding_file)
            # Filter embeddings to match the trials included for a given subject
            X_embeddings_mapping = {k: v for k, v in zip(nsd_vg_metadata["nsdId"], X_embeddings_initial)}
            # Get a list with a subset of the embeddings based on those that appear
            # in trial_info_short.nsdId, and concatenate them into a 2D array
            X_embeddings = np.array([X_embeddings_mapping[nsd_id] for nsd_id in trial_info_short["nsdId"]])

            # Fit encoder models
            encoder = RidgeCV()
            encoder.fit(X_embeddings, Y)

            # Predicted responses (transpose back to 4D shape)
            Y_pred = encoder.predict(X_embeddings).T

            # TODO evaluate predictions with 2v2 and pearson correlation
            logger.info(f"Subject {subject} encoding process complete.")


if __name__ == "__main__":
    run_neural_encoder()

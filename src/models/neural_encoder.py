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
from scipy.spatial.distance import cosine
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore", module="sklearn")


def pairwise_accuracy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Compute a 2v2 pairwise accuracy between predicted responses (Y_pred)
    and true responses (Y_true). For each pair (i, j), check whether
    sim(Y_pred[i], Y_true[i]) + sim(Y_pred[j], Y_true[j])
    is greater than sim(Y_pred[i], Y_true[j]) + sim(Y_pred[j], Y_true[i]).

    Args:
        Y_pred (np.ndarray): (#samples, #features) array of predicted responses.
        Y_true (np.ndarray): (#samples, #features) array of true responses.

    Returns:
        float: The fraction of correct 2v2 comparisons.
    """
    correct = 0
    total = 0
    for i in range(len(Y_pred)):
        for j in range(i + 1, len(Y_pred)):
            sim_ii = 1 - cosine(Y_pred[i], Y_true[i])
            sim_jj = 1 - cosine(Y_pred[j], Y_true[j])
            sim_ij = 1 - cosine(Y_pred[i], Y_true[j])
            sim_ji = 1 - cosine(Y_pred[j], Y_true[i])
            if sim_ii + sim_jj > sim_ij + sim_ji:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def pearson_correlation(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Compute the mean Pearson correlation across features (voxels) for
    predicted vs. true responses.

    Args:
        Y_pred (np.ndarray): (#samples, #features) array of predicted responses.
        Y_true (np.ndarray): (#samples, #features) array of true responses.

    Returns:
        float: The average correlation across features.
    """
    correlations = []
    for feature_idx in range(Y_pred.shape[1]):
        corr_matrix = np.corrcoef(Y_pred[:, feature_idx], Y_true[:, feature_idx])
        correlations.append(corr_matrix[0, 1])
    return float(np.nanmean(correlations))


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

    # Create a results df
    results = pd.DataFrame(
        columns=[
            "subject",
            "model_id",
            "pairwise_accuracy_mean",
            "pairwise_accuracy_std",
            "pearson_correlation_mean",
            "pearson_correlation_std",
        ]
    )

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

        # Average betas by cocoId
        unique_coco_ids = trial_info_short["cocoId"].unique()
        Y_aggregated_list = []
        for c_id in unique_coco_ids:
            idx = trial_info_short.index[trial_info_short["cocoId"] == c_id].tolist()
            Y_agg = Y[idx].mean(axis=0)
            Y_aggregated_list.append(Y_agg)
        Y_aggregated = np.stack(Y_aggregated_list, axis=0)  # (#unique_coco_ids, #voxels)

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

            # Average embeddings for trials sharing the same cocoId
            X_aggregated_list = []
            for c_id in unique_coco_ids:
                idx = trial_info_short.index[trial_info_short["cocoId"] == c_id].tolist()
                X_agg = X_embeddings[idx].mean(axis=0)
                X_aggregated_list.append(X_agg)
            X_aggregated = np.stack(X_aggregated_list, axis=0)  # (#unique_coco_ids, embedding_dim)

            # 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            pairwise_accs = []
            pearson_corrs = []

            for train_idx, test_idx in kf.split(X_aggregated):
                encoder = RidgeCV()
                encoder.fit(X_aggregated[train_idx], Y_aggregated[train_idx])

                # Evaluate predictions with 2v2 and pearson correlation
                Y_pred = encoder.predict(X_aggregated[test_idx])
                pairwise_accs += [pairwise_accuracy(Y_pred, Y_aggregated[test_idx])]
                pearson_corrs += [pearson_correlation(Y_pred, Y_aggregated[test_idx])]

            results = results.append(
                {
                    "subject": subject,
                    "model_id": model_id,
                    "pairwise_accuracy_mean": np.mean(pairwise_accs),
                    "pairwise_accuracy_std": np.std(pairwise_accs),
                    "pearson_correlation_mean": np.mean(pearson_corrs),
                    "pearson_correlation_std": np.std(pearson_corrs),
                },
                ignore_index=True,
            )

        logger.info(f"Subject {subject} encoding process complete.")


if __name__ == "__main__":
    run_neural_encoder()

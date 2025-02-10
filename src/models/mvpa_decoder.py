"""MVPA searchlight decoder for NSD data."""

import logging
import os
import warnings
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nilearn.decoding import SearchLight
from nsd_access import NSDAccess
from omegaconf import DictConfig
from scipy.stats import zscore

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore", module="sklearn")


@hydra.main(config_path="../../configs/model", config_name="mvpa_decoder")
def run_searchlight_decoder(cfg: DictConfig) -> None:
    """
    Run MVPA searchlight decoder on NSD data to predict a target variable from the NSD-COCO overlap.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra.
    """
    # Load NSD data
    nsd_dir = Path(cfg.data.large_data_path) / cfg.data.nsd_directory
    nsd = NSDAccess(nsd_dir)

    # Load NSD-VG metadata
    nsd_vg_metadata = pd.read_csv(nsd_dir / "nsd_vg" / "nsd_vg_metadata.csv")

    # Process each subject
    for subject in cfg.subjects:
        logger.info(f"Processing subject {subject}")

        # Get betas for all sessions for the subject
        all_betas = []
        for session in range(1, cfg.max_sessions + 1):
            session_betas = nsd.read_betas(
                subject, session_index=session, data_format=cfg.data.data_format, data_type=cfg.data.data_type
            )
            # z-scoring of session-betas along the last axis (i.e., for each voxel across trials within a session)
            session_betas = zscore(session_betas, axis=-1)
            # In full float64 precision, each session requires 3.9GB of memory,
            # so instead: cast to smaller data type (0.98GB)
            session_betas = session_betas.astype(np.float16)
            # Replace NaNs with 0s (that may result from dividing by 0)
            # If there is no variance in a voxel across trials, then we're not interested in it anyways
            session_betas = np.nan_to_num(session_betas)
            all_betas.append(session_betas)

        # Concatenate all betas
        betas = np.concatenate(all_betas, axis=-1)
        # Get affine and header from NSD
        affine, header = nsd.affine_header(subject, data_format=cfg.data.data_format)

        # Create mask of non-zero voxels (across all trials)
        brain_mask = (np.abs(betas).sum(axis=-1) > 0).astype(np.int32)

        # Get trial info for the subject based on nsd_vg_metadata
        subject_num = subject.replace("subj0", "subject")
        trial_info = nsd_vg_metadata[nsd_vg_metadata[subject_num] > 0]

        # Instead of three separate rep columns, the dataframe should become a long
        # one with a column for all reps
        trial_info = trial_info.melt(
            id_vars=[col for col in trial_info.columns if "subject" not in col],
            value_vars=[f"{subject_num}_rep0", f"{subject_num}_rep1", f"{subject_num}_rep2"],
            var_name="rep",
            value_name="trial_index",
        )

        # Remove all entries in the trial_index column larger than the number of
        # trials (betas.shape[-1])
        trial_info_short = trial_info[trial_info["trial_index"] < betas.shape[-1]]
        # Sort by trial_index
        trial_info_short = trial_info_short.sort_values(by="trial_index").reset_index(drop=True)

        # Extract features and apply mask
        X = betas[:, :, :, trial_info_short["trial_index"].values - 1]

        # Convert X and mask to nifti for searchlight
        X = nib.Nifti1Image(X, affine=affine, header=header)
        mask_img = nib.Nifti1Image(brain_mask, affine=affine, header=header)
        # Use the nsdgeneral mask
        if cfg.nsdgeneral_mask:
            # Use a much smaller mask in the subject native space
            # Read the atlas results for the given subject
            atlas_results = nsd.read_atlas_results(subject, data_format=cfg.data.data_format, atlas="nsdgeneral")[0]
            # Set all -1 values to 0
            atlas_results[atlas_results == -1] = 0
            # Create a nifti image from the atlas results
            mask_img = nib.Nifti1Image(atlas_results, affine=affine, header=header)

        for target_variable in cfg.target_variables:
            # Define the target variable
            if cfg.binarize_target:
                # Binarize the target into "high" and "low" depending on half between min and max
                median = np.percentile(trial_info_short[target_variable], 50)
                trial_info_short[target_variable] = (trial_info_short[target_variable] > median).astype(int)
            y = trial_info_short[target_variable].values

            # Configure searchlight with mask
            searchlight = SearchLight(
                mask_img=mask_img,
                radius=cfg.searchlight.radius,
                estimator=cfg.searchlight.estimator,
                n_jobs=cfg.searchlight.n_jobs,
                scoring=cfg.searchlight.scoring,
                verbose=1,
                cv=cfg.searchlight.cv,  # 5-fold cross-validation instead of train/test split
            )

            # Fit searchlight
            searchlight.fit(X, y)

            # Save results
            output_path = Path(cfg.data.output_dir) / f"{subject}_{target_variable}_searchlight_map.nii.gz"
            nib.save(searchlight.scores_img_, output_path)

            logger.info(
                f"Subject {subject}, {target_variable} cv {cfg.searchlight.scoring}: "
                f"{searchlight.scores_.mean():.3f}"
            )

    logger.info("MVPA searchlight decoding complete.")


if __name__ == "__main__":
    run_searchlight_decoder()

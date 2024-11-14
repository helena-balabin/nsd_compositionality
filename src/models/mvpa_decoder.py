"""MVPA decoder for NSD data using nilearn."""

import logging
import os
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nilearn.decoding import Decoder
from nsd_access import NSDAccess
from omegaconf import DictConfig
from scipy.stats import zscore

logger = logging.getLogger(__name__)
load_dotenv()


@hydra.main(config_path="../../configs/model", config_name="mvpa_decoder")
def run_mvpa_decoder(cfg: DictConfig) -> None:
    """
    Run MVPA decoder on NSD data to predict a target variable from the NSD-COCO overlap.

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
            # Replace NaNs with 0s (that may result from dividing by 0)
            # If there is no variance in a voxel across trials, then we're not interested in it anyways
            session_betas = np.nan_to_num(session_betas)
            all_betas.append(session_betas)

        # Concatenate all betas
        betas = np.concatenate(all_betas, axis=-1)

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

        # Extract features (betas) and target
        # Index the betas with the trial_index (trial_index starts at 1)
        # Define train and test set
        X = betas[:, :, :, trial_info_short["trial_index"].values - 1]

        # Use the "shared 1000" as the validation set, and the rest as the training set
        X_train = X[:, :, :, trial_info_short["shared1000"] == False]  # noqa: E712
        X_test = X[:, :, :, trial_info_short["shared1000"] == True]  # noqa: E712
        # Log the number of examples for this subject
        logger.info(f"Number of examples for subject {subject}: {X.shape[-1]}")

        # Get the correct affine and header for this subject
        affine, header = nsd.affine_header(subject, data_format=cfg.data.data_format)
        X_train_nifti = nib.Nifti1Image(X_train, affine=affine, header=header)
        X_test_nifti = nib.Nifti1Image(X_test, affine=affine, header=header)

        for target_variable in cfg.target_variables:
            # Define the target variable
            if cfg.binarize_target:
                # Binarize the target into "high" and "low" depending on half between min and max
                median = np.percentile(trial_info_short[target_variable], 50)
                trial_info_short[target_variable] = (trial_info_short[target_variable] > median).astype(int)
            y = trial_info_short[target_variable].values
            y_train = y[trial_info_short["shared1000"] == False]  # noqa: E712
            y_test = y[trial_info_short["shared1000"] == True]  # noqa: E712

            # Don't use a mask in order to do whole-brain decoding
            decoder = Decoder(
                estimator=cfg.decoder.estimator,
                scoring=cfg.decoder.scoring,
                screening_percentile=cfg.decoder.screening_percentile,
                standardize=False,  # we've already z-scored the data per session
                cv=None,
            )
            decoder.fit(X_train_nifti, y_train)
            score = decoder.score(X_test_nifti, y_test)

            logger.info(
                f"Decoder {cfg.decoder.scoring} score for subject {subject} and "
                f"target variable {target_variable} on the test set: {score}"
            )

            # Save the coef_img_ for the weights from the training data and positive class
            nib.save(
                decoder.coef_img_[1],
                os.path.join(cfg.data.output_dir, f"decoder_weights_{subject}_{target_variable}_1.nii.gz"),
            )

    logger.info("MVPA decoding complete.")


if __name__ == "__main__":
    run_mvpa_decoder()

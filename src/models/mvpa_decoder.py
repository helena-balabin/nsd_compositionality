"""MVPA decoder for NSD data using nilearn."""

import os
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from nilearn import plotting
from nilearn.decoding import Decoder
from nsd_access import NSDAccess
from omegaconf import DictConfig

load_dotenv()


@hydra.main(config_path="../../configs/model", config_name="mvpa_decoder")
def run_mvpa_decoder(cfg: DictConfig) -> None:
    """
    Run MVPA decoder on NSD data to predict 'sg_depth'.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra.
    """
    # Load NSD data
    nsd_dir = Path(cfg.data.large_data_path) / cfg.data.nsd_directory
    nsd = NSDAccess(nsd_dir)

    # Load NSD-VG metadata
    nsd_vg_metadata = pd.read_csv(nsd_dir / "nsd_vg" / "nsd_vg_metadata.csv")
    if cfg.binarize_target:
        # Binarize sg_depth into "high" and "low" depending on half between min and max
        min_sg_depth = nsd_vg_metadata["sg_depth"].min()
        max_sg_depth = nsd_vg_metadata["sg_depth"].max()
        median_sg_depth = (min_sg_depth + max_sg_depth) / 2
        nsd_vg_metadata["sg_depth"] = (nsd_vg_metadata["sg_depth"] > median_sg_depth).astype(int)

    # Process each subject
    for subject in cfg.subjects:
        logger.info(f"Processing subject {subject}")

        # Load the "nsdgeneral" mask for a given subject
        mask_raw = nib.load(
            nsd_dir / "nsddata" / "ppdata" / f"{subject}" / f"{cfg.data.data_format}" / "roi" / "nsdgeneral.nii.gz"
        )
        # Replace -1 with 0 in the mask
        mask = mask_raw.get_fdata()
        mask[mask == -1] = 0
        mask = nib.Nifti1Image(mask, affine=mask_raw.affine, header=mask_raw.header)

        # Get betas for all sessions for the subject
        all_betas = []
        for session in range(1, cfg.max_sessions + 1):
            session_betas = nsd.read_betas(
                subject, session_index=session, data_format=cfg.data.data_format, data_type=cfg.data.data_type
            )
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

        # Extract features (betas) and target (sg_depth)
        # Index the betas with the trial_index (trial_index starts at 1)
        X = betas[:, :, :, trial_info_short["trial_index"].values - 1]
        y = trial_info_short["sg_depth"].values
        # Log the number of examples for this subject
        logger.info(f"Number of examples for subject {subject}: {X.shape[0]}")

        # Get the correct affine and header for this subject
        affine, header = nsd.affine_header(subject, data_format=cfg.data.data_format)
        X_nifti = nib.Nifti1Image(X, affine=affine, header=header)

        # TODO make sure stimuli are not repeated in training and testing
        # Initialize and run the decoder with cross-validation
        decoder = Decoder(
            estimator=cfg.decoder.estimator,
            cv=cfg.decoder.cv,
            scoring="roc_auc",
            mask=mask,
        )
        decoder.fit(X_nifti, y)

        # Get cross-validation scores
        cv_scores = decoder.cv_scores_

        # Save the coef_img_ of the positive class
        nib.save(decoder.coef_img_[1], os.path.join(cfg.data.output_dir, f"decoder_weights_{subject}_1.nii.gz"))

        # For each class in the cv_scores dict:
        for class_name, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(
                f"Cross-validation scores for subject {subject} and class {class_name}: "
                f"{mean_score:.3f} ± {std_score:.3f}"
            )
            # Generate a plot with the weights of the decoder and save it
            plot = plotting.view_img(decoder.coef_img_[class_name], title=f"Decoder weights for {class_name}", dim=-1)
            # Save the plot as a html file
            plot.save_as_html(f"decoder_weights_{subject}_{class_name}.html")

    logger.info("MVPA decoding complete.")


if __name__ == "__main__":
    run_mvpa_decoder()

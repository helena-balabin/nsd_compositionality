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
from himalaya.ridge import Ridge
from nilearn.decoding import Decoder, SearchLight
from nsd_access import NSDAccess
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold

from nsd_compositionality.utils.nsd_data_utils import (
    get_trial_info_for_subject,
    load_betas_original_method,
    load_or_create_cached_betas,
)

os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
load_dotenv()
warnings.filterwarnings("ignore", module="sklearn")


@hydra.main(config_path="../../../configs/model", config_name="mvpa_decoder")
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

    # Initialize results DataFrame
    results = pd.DataFrame(
        columns=[
            "subject",
            "target_variable",
            f"{cfg.classifier.scoring}_mean",
            f"{cfg.classifier.scoring}_std",
        ]
    )

    # Process each subject
    for subject in cfg.subjects:
        logger.info(f"Processing subject {subject}")

        # Load or create cached betas
        if cfg.data.use_cached_betas:
            betas, brain_mask, metadata = load_or_create_cached_betas(
                nsd=nsd,
                subject=subject,
                max_sessions=cfg.max_sessions,
                data_format=cfg.data.data_format,
                data_type=cfg.data.data_type,
                cache_dir=Path(cfg.data.betas_cache_dir),
                force_reload=cfg.data.force_reload_cache,
            )
            # Get affine and header from metadata
            affine = np.array(metadata["affine"])
            header = nsd.affine_header(subject, data_format=cfg.data.data_format)[1]
            successful_sessions = metadata.get("successful_sessions", None)
        else:
            # Original data loading method (fallback)
            betas, affine, header, brain_mask = load_betas_original_method(
                nsd=nsd,
                subject=subject,
                max_sessions=cfg.max_sessions,
                data_format=cfg.data.data_format,
                data_type=cfg.data.data_type,
            )
            successful_sessions = None  # Not tracked in original method

        # Get trial info for the subject using the utility function
        trial_info_short = get_trial_info_for_subject(nsd_vg_metadata, subject, betas.shape, successful_sessions)

        # Validate trial indices before extraction
        max_trial_index = trial_info_short["trial_index"].max()
        if max_trial_index >= betas.shape[-1]:
            logger.error(
                f"Subject {subject}: Maximum trial index ({max_trial_index}) exceeds "
                f"available trials ({betas.shape[-1]}). Check session loading."
            )
            raise ValueError(f"Trial index mismatch for {subject}")

        # Extract features and apply mask
        X = betas[:, :, :, trial_info_short["trial_index"].values - 1]

        logger.info(f"Subject {subject}: Extracted {X.shape[-1]} trials from betas with shape {betas.shape}")

        # Convert data to nifti for searchlight and nilearn decoder
        X = nib.Nifti1Image(X, affine=affine, header=header)
        # Create a whole-brain mask as a start
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
        elif cfg.snr_mask:
            # Use the SNR mask
            snr_mask_path = (
                Path(cfg.data.large_data_path)
                / cfg.data.nsd_directory
                / "nsddata_betas"
                / "ppdata"
                / subject
                / cfg.data.data_format
                / cfg.data.data_type
                / "ncsnr.nii.gz"
            )
            snr_mask_img = nib.load(snr_mask_path)
            # Convert SNR values to actual noise ceilings
            snr_mask_data = snr_mask_img.get_fdata()
            # Use the formula from the NSD data manual
            nc_mask_data = snr_mask_data**2 / (snr_mask_data**2 + 1 / 3)  # Convert SNR to noise ceiling
            # Threshold the SNR mask
            nc_mask_data[nc_mask_data < cfg.snr_mask_threshold] = 0
            nc_mask_data[nc_mask_data >= cfg.snr_mask_threshold] = 1
            mask_img = nib.Nifti1Image(nc_mask_data, affine=snr_mask_img.affine, header=snr_mask_img.header)

        # Process each target variable (graph measure)
        for target_variable in cfg.target_variables:
            # Define the target variable
            if cfg.binarize_target:
                # Binarize the target into "high" and "low" depending on half between min and max
                median = np.percentile(trial_info_short[target_variable], 50)
                trial_info_short[target_variable] = (trial_info_short[target_variable] > median).astype(int)
            y = trial_info_short[target_variable].values

            # Use cocoId as groups to ensure repetitions of the same stimulus stay together
            groups = trial_info_short["cocoId"].values

            # Create a list of train/test splits using GroupKFold
            group_cv = StratifiedGroupKFold(n_splits=cfg.classifier.cv)
            cv_splits = list(group_cv.split(X=np.zeros(len(y)), y=y, groups=groups))

            # Use either a searchlight or a classic decoder on all voxels
            # Note that the classifier can also be a regressor
            if cfg.use_searchlight:
                classifier = SearchLight(
                    mask_img=mask_img,
                    radius=cfg.searchlight.radius,
                    estimator=Ridge() if cfg.classifier.estimator == "ridge" else cfg.classifier.estimator,
                    n_jobs=cfg.classifier.n_jobs,
                    scoring=cfg.classifier.scoring,
                    cv=cv_splits,
                )
            else:
                classifier = Decoder(
                    mask=mask_img,
                    estimator=Ridge() if cfg.classifier.estimator == "ridge" else cfg.classifier.estimator,
                    n_jobs=cfg.classifier.n_jobs,
                    scoring=cfg.classifier.scoring,
                    cv=cv_splits,
                )

            # Fit searchlight or decoder
            # Note: nilearn handles cross-validation internally using the provided cv_splits
            classifier.fit(X, y)
            # Write results to the DataFrame
            if cfg.use_searchlight:
                mean = np.mean(classifier.masked_scores_)
                std = np.std(classifier.masked_scores_)
            else:
                mean = np.mean(classifier.cv_scores_[1])
                std = np.std(classifier.cv_scores_[1])
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        [
                            {
                                "subject": subject,
                                "target_variable": target_variable,
                                f"{cfg.classifier.scoring}_mean": mean,
                                f"{cfg.classifier.scoring}_std": std,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            # Save results
            if cfg.use_searchlight:
                searchlight_img_output_path = (
                    Path(cfg.data.output_dir) / f"{subject}_{target_variable}_searchlight_map.nii.gz"
                )
                nib.save(classifier.scores_img_, searchlight_img_output_path)

    if cfg.use_searchlight:
        output_path = Path(cfg.data.output_dir) / "mvpa_searchlight_results.csv"
    else:
        output_path = Path(cfg.data.output_dir) / "mvpa_decoder_results.csv"
    results.to_csv(output_path, index=False)

    logger.info("MVPA complete.")


if __name__ == "__main__":
    run_searchlight_decoder()

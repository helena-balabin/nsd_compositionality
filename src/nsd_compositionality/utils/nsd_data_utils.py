"""Utility functions for loading and preprocessing NSD data."""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import zscore

logger = logging.getLogger(__name__)


def safe_zscore(data, axis=-1, epsilon=1e-9):
    """Calculate z-score, handling near-zero standard deviation and NaNs.

    This function suppresses runtime warnings for division by zero, as these
    cases are handled explicitly by the `np.where` condition.

    Args:
        data (np.ndarray): Input data.
        axis (int): The axis along which to operate.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Z-scored data.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.nanmean(data, axis=axis, keepdims=True)
        std = np.nanstd(data, axis=axis, keepdims=True)

        # Where std is close to zero or NaN, z-score should be 0.
        # The division is evaluated before the where, causing a benign warning.
        # We suppress the warning and let np.where select the correct value.
        z_scored_data = np.where(np.isnan(std) | (std < epsilon), 0.0, (data - mean) / std)

    return z_scored_data


def load_or_create_cached_betas(nsd, subject, max_sessions, data_format, data_type, cache_dir, force_reload=False):
    """Load cached preprocessed betas or create them if they don't exist.

    Args:
        nsd: NSDAccess instance
        subject: Subject identifier (e.g., 'subj01')
        max_sessions: Maximum number of sessions to load
        data_format: Data format (e.g., 'func1pt8mm')
        data_type: Data type (e.g., 'betas_fithrf_GLMdenoise_RR')
        cache_dir: Directory to store cached files
        force_reload: Whether to force reloading even if cache exists

    Returns:
        tuple: (betas, metadata)
    """
    cache_dir = Path(cache_dir) / "nsd_betas_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{subject}_{data_format}_{data_type}_sessions{max_sessions}_betas.npy"
    cache_path = cache_dir / filename
    metadata_path = cache_path.with_suffix(".json")

    # Check if cached data exists and is valid
    if cache_path.exists() and metadata_path.exists() and not force_reload:
        logger.info(f"Loading cached betas for {subject} from {cache_path}")
        try:
            betas = np.load(cache_path)

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            logger.info(f"Loaded cached betas with shape {betas.shape}")
            return betas, metadata

        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}. Regenerating...")

    logger.info(f"Creating cached betas for {subject} (this may take a while...)")

    # Load and preprocess all sessions
    all_betas = []
    successful_sessions = []

    for session in range(1, max_sessions + 1):
        try:
            session_betas = nsd.read_betas(
                subject,
                session_index=session,
                data_format=data_format,
                data_type=data_type,
            )

            # Each session has exactly 12 runs, with 63 trials for odd runs and 62 for even runs
            total_trials = session_betas.shape[-1]
            n_runs = 12  # Fixed number of runs

            # Define trials per run: odd runs (1,3,5,7,9,11) = 63, even runs (2,4,6,8,10,12) = 62
            trials_per_run = [63 if (run_idx + 1) % 2 == 1 else 62 for run_idx in range(n_runs)]

            # Z-score each run separately with correct indexing
            current_idx = 0
            for run_idx in range(n_runs):
                run_trials = trials_per_run[run_idx]
                start_idx = current_idx
                end_idx = current_idx + run_trials

                # Extract the run data
                run_data = session_betas[:, :, :, start_idx:end_idx]

                # Z-score within this run
                run_data_zscore = safe_zscore(run_data, axis=-1)

                # Put it back
                session_betas[:, :, :, start_idx:end_idx] = run_data_zscore

                # Update current index for next run
                current_idx = end_idx

            session_betas = session_betas.astype(np.float16)

            all_betas.append(session_betas)
            successful_sessions.append(session)

            logger.info(f"Processed session {session} for {subject}")

        except Exception as e:
            logger.warning(f"Failed to load session {session} for {subject}: {e}")
            continue

    if not all_betas:
        raise ValueError(f"No sessions could be loaded for {subject}")

    # Log session loading summary
    loaded_sessions = len(successful_sessions)
    total_trials = sum(beta.shape[-1] for beta in all_betas)
    logger.info(
        f"Subject {subject}: Successfully loaded {loaded_sessions}/{max_sessions} sessions "
        f"with {total_trials} total trials"
    )

    if loaded_sessions < max_sessions:
        missing_sessions = set(range(1, max_sessions + 1)) - set(successful_sessions)
        logger.warning(f"Subject {subject}: Missing sessions {sorted(missing_sessions)}")

    # Concatenate all successful sessions
    betas = np.concatenate(all_betas, axis=-1)
    # Get affine and header from NSD
    affine, _ = nsd.affine_header(subject, data_format=data_format)

    # Prepare metadata
    metadata = {
        "subject": subject,
        "data_format": data_format,
        "data_type": data_type,
        "max_sessions": max_sessions,
        "successful_sessions": successful_sessions,
        "shape": betas.shape,
        "dtype": str(betas.dtype),
        "affine": affine.tolist(),
        "preprocessing": {
            "zscore_axis": -1,
            "dtype_conversion": "float16",
            "nan_handling": "nan_to_num",
        },
    }

    # Save cached data
    logger.info(f"Saving cached betas to {cache_path}")
    np.save(cache_path, betas)

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created cached betas with shape {betas.shape}")
    return betas, metadata


def load_betas_original_method(nsd, subject, max_sessions, data_format, data_type):
    """Load betas using the original method (fallback when caching is disabled).

    Args:
        nsd: NSDAccess instance
        subject: Subject identifier
        max_sessions: Maximum number of sessions to load
        data_format: Data format
        data_type: Data type

    Returns:
        tuple: (betas, affine, header)
    """
    logger.info("Using original data loading (caching disabled)")
    all_betas = []

    for session in range(1, max_sessions + 1):
        try:
            session_betas = nsd.read_betas(
                subject,
                session_index=session,
                data_format=data_format,
                data_type=data_type,
            )
            # z-scoring of session-betas for each run
            # We know that there are 12 runs per session
            trials_per_run = session_betas.shape[-1] // 12
            # Z-score within each run
            session_betas = zscore(
                session_betas.reshape(
                    session_betas.shape[0],
                    session_betas.shape[1],
                    session_betas.shape[2],
                    trials_per_run,
                ),
                axis=-1,
            )
            # Reshape back to original shape
            session_betas = session_betas.reshape(
                session_betas.shape[0],
                session_betas.shape[1],
                session_betas.shape[2],
                -1,  # All trials
            )

            # In full float64 precision, each session requires 3.9GB of memory,
            # so instead: cast to smaller data type (0.98GB)
            session_betas = session_betas.astype(np.float16)
            # Replace NaNs with 0s (that may result from dividing by 0)
            # If there is no variance in a voxel across trials, then we're not interested in it anyways
            session_betas = np.nan_to_num(session_betas)
            all_betas.append(session_betas)

        except Exception as e:
            logger.warning(f"Failed to load session {session} for {subject}: {e}")
            continue

    if not all_betas:
        raise ValueError(f"No sessions could be loaded for {subject}")

    # Log session loading summary (for original method)
    loaded_sessions = len(all_betas)
    total_trials = sum(beta.shape[-1] for beta in all_betas)
    logger.info(
        f"Subject {subject}: Successfully loaded {loaded_sessions}/{max_sessions} sessions "
        f"with {total_trials} total trials (original method)"
    )

    if loaded_sessions < max_sessions:
        logger.warning(f"Subject {subject}: Some sessions may be missing (original method doesn't track which)")

    # Concatenate all betas
    betas = np.concatenate(all_betas, axis=-1)
    # Get affine and header from NSD
    affine, header = nsd.affine_header(subject, data_format=data_format)

    return betas, affine, header


def get_trial_info_for_subject(nsd_vg_metadata, subject, betas_shape, successful_sessions=None):
    """Get trial information for a specific subject.

    Args:
        nsd_vg_metadata: NSD VG metadata DataFrame
        subject: Subject identifier (e.g., 'subj01')
        betas_shape: Shape of the betas array
        successful_sessions: List of successfully loaded sessions (optional)

    Returns:
        pd.DataFrame: Trial information for the subject
    """
    subject_num = subject.replace("subj0", "subject")
    trial_info = nsd_vg_metadata[nsd_vg_metadata[subject_num] > 0]

    # Reshape the dataframe to have one row per trial
    trial_info = trial_info.melt(
        id_vars=[col for col in trial_info.columns if "subject" not in col],
        value_vars=[
            f"{subject_num}_rep0",
            f"{subject_num}_rep1",
            f"{subject_num}_rep2",
        ],
        var_name="rep",
        value_name="trial_index",
    )

    # Filter out invalid trial indices based on actual betas shape
    trial_info_short = trial_info[trial_info["trial_index"] < betas_shape[-1]]

    # Additional validation if we have session information
    if successful_sessions is not None:
        # Log session information
        total_expected_sessions = max(successful_sessions) if successful_sessions else 0
        missing_sessions = set(range(1, total_expected_sessions + 1)) - set(successful_sessions)

        if missing_sessions:
            logger.warning(
                f"Subject {subject}: Missing sessions {sorted(missing_sessions)}. "
                f"Trial indices may be shifted. Loaded sessions: {sorted(successful_sessions)}"
            )

        logger.info(
            f"Subject {subject}: Using {len(successful_sessions)} sessions out of "
            f"{total_expected_sessions} expected. Final trial count: {len(trial_info_short)}"
        )

    trial_info_short = trial_info_short.sort_values(by="trial_index").reset_index(drop=True)

    return trial_info_short

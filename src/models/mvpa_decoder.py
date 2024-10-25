"""MVPA decoder for NSD data using nilearn."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from nilearn.decoding import Decoder
from nsd_access import NSDAccess
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

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

    # Process each subject
    for subject in cfg.subjects:
        logger.info(f"Processing subject {subject}")

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

        # TODO get the nibabel format working
        # https://nilearn.github.io/dev/auto_examples/00_tutorials/
        # plot_decoding_tutorial.html#sphx-glr-auto-examples-00-tutorials-plot-decoding-tutorial-py
        # TODO make sure that the same cocoId is not present in both train and test
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state
        )

        # Initialize and run the decoder
        decoder = Decoder(estimator=cfg.decoder.estimator, cv=cfg.decoder.cv)
        decoder.fit(X_train, y_train)

        # Evaluate the decoder
        score = decoder.score(X_test, y_test)
        logger.info(f"Decoder score for subject {subject}: {score}")

    logger.info("MVPA decoding complete.")


if __name__ == "__main__":
    run_mvpa_decoder()

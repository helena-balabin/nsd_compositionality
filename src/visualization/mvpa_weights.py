"""Visualization of MVPA decoder weights using pycortex."""

from pathlib import Path

import hydra
import nibabel as nib
from loguru import logger
from nilearn import plotting
from omegaconf import DictConfig


@hydra.main(config_path="../../configs/visualization", config_name="mvpa_decoder_weights")
def run_visualization(cfg: DictConfig) -> None:
    """
    Create pycortex visualizations of decoder weights.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load decoder weights
    weight_files = list(Path(cfg.data.input_dir).glob("*weights*.nii.gz"))
    logger.info(f"Found {len(weight_files)} weight files")

    for weight_file in weight_files:
        logger.info(f"Processing {weight_file.name}")

        # Load weights
        weights_img = nib.load(weight_file)
        subject_index = int(weight_file.stem.split("_")[-2].strip("subj0"))

        # nsd/nsddata_betas/ppdata/subj0X/func1pt8mm/
        # betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz as background image
        background_img = nib.load(
            Path(cfg.data.nsd_directory) / f"nsddata_betas/ppdata/subj0{subject_index}/"
            f"{cfg.mapping.space_from}/{cfg.mapping.betas_type}/betas_session01.nii.gz"
        )
        # But only take the first volume
        background_img = nib.Nifti1Image(background_img.get_fdata()[:, :, :, 0], background_img.affine)

        # Plot
        display = plotting.plot_stat_map(
            weights_img,
            cmap=cfg.visualization.colormap,
            bg_img=background_img,
        )
        display.savefig(output_dir / f"weights_subj{subject_index}.png")
        logger.info(f"Saved weights visualization to {output_dir / f'weights_subj{subject_index}.png'}")


if __name__ == "__main__":
    run_visualization()

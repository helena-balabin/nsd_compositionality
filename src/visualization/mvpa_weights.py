"""Visualization of MVPA decoder weights using pycortex."""

import logging
from pathlib import Path

import hydra
from nilearn.experimental.surface import PolyData, SurfaceImage, load_fsaverage, load_fsaverage_data
from nilearn.plotting import plot_surf_stat_map
from nsdcode import NSDmapdata
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


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

        # Get subject index from filename
        subject_index = int(weight_file.stem.split("_")[-2].strip("subj0"))

        # Map from functional data to cortical surface
        nsd_mapper = NSDmapdata(cfg.data.nsd_directory)

        weights_img_list = {}

        for hemisphere in ["right", "left"]:
            # 1. Map from functional to "white"
            weights_img = nsd_mapper.fit(
                subject_index,
                cfg.mapping.space_from,
                f"{hemisphere[0]}h.white",
                str(weight_file),
            )
            # 2. Map from "white" to "fsaverage"
            weights_img = nsd_mapper.fit(
                subject_index,
                f"{hemisphere[0]}h.white",
                cfg.mapping.space_to,
                weights_img,
            )
            weights_img_list[hemisphere] = weights_img

        # Load the background mesh
        big_fsaverage_meshes = load_fsaverage("fsaverage")
        big_fsaverage_sulcal = load_fsaverage_data(mesh_name="fsaverage", data_type="sulcal", mesh_type="inflated")

        # Convert the projected beta data to the right format
        data = PolyData(right=weights_img_list["right"], left=weights_img_list["left"])
        big_img = SurfaceImage(
            mesh=big_fsaverage_meshes["pial"],  # Pial vs inflated?
            data=data,
        )

        # Generate the plot
        plot_surf_stat_map(
            stat_map=big_img,
            surf_mesh=big_fsaverage_meshes["inflated"],
            colorbar=True,
            title="Surface fine mesh",
            bg_map=big_fsaverage_sulcal,
            threshold=cfg.visualization.threshold,
            output_file=output_dir / f"weights_subj{subject_index}_surface.png",
            cmap=cfg.visualization.cmap,
        )

        logger.info(f"Saved weights visualization to {output_dir / f'weights_subj{subject_index}_surface.png'}")


if __name__ == "__main__":
    run_visualization()

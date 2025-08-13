"""Visualization of MVPA decoder weights using pycortex."""

import logging
import re
from pathlib import Path

import hydra
import nibabel as nib
from dotenv import load_dotenv
from nilearn.datasets import load_fsaverage, load_fsaverage_data, load_mni152_template
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import SurfaceImage
from nsdcode import NSDmapdata
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

load_dotenv()


@hydra.main(config_path="../../../configs/visualization", config_name="mvpa_decoder_weights")
def run_visualization(cfg: DictConfig) -> None:
    """
    Create nilearn visualizations of decoder-based accuracy maps.

    Args:
        cfg (DictConfig): The configuration object loaded by Hydra.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all target variables
    for target_variable in cfg.target_variables:
        # Load decoder weights
        weight_files = list(Path(cfg.data.input_dir).glob(f"subj*_{target_variable}*searchlight*.nii.gz"))
        logger.info(f"Found {len(weight_files)} weight files")

        for weight_file in weight_files:
            logger.info(f"Processing {weight_file.name}")

            # Get subject index from filename
            # Get the "subjXX" part of the filename with a regex
            subject_index = int(re.search(r"subj\d+", weight_file.name).group(0).replace("subj", ""))  # type: ignore

            # Map from functional data to cortical surface
            nsd_mapper = NSDmapdata(cfg.data.nsd_directory)

            # Map from functional to MNI
            weights_img = nsd_mapper.fit(
                subject_index,
                cfg.mapping.space_from,
                cfg.mapping.space_to,
                str(weight_file),
                badval=0,
            )
            # The resulting variable is in RPI format, see the following:
            # "When using nsd_mapdata to map to MNI space, note that the output variable is
            # returned to the workspace in RPI ordering. But notice that if you ask nsd_mapdata
            # to write out a NIFTI file, that file has data stored in LPI ordering."
            # Convert weights_img (a 3D numpy array) to an niimg-like object
            mni = load_mni152_template()
            weights_img = nib.Nifti1Image(weights_img, mni.affine)

            # Load fsaverage
            fsaverage = load_fsaverage("fsaverage")
            fsaverage_sulcal = load_fsaverage_data(
                mesh="fsaverage",
                data_type="sulcal",
                mesh_type="inflated",
            )
            # Get surface from MNI projection
            surface_image = SurfaceImage.from_volume(
                volume_img=weights_img,
                mesh=fsaverage["pial"],
            )

            # Generate the output file path
            output_file = output_dir / weight_file.with_name(weight_file.stem.split(".")[0] + "_surface.png").name
            # Generate the plot
            plot_surf_stat_map(
                stat_map=surface_image,
                surf_mesh=fsaverage["inflated"],
                colorbar=True,
                title="Surface fine mesh",
                bg_map=fsaverage_sulcal,
                threshold=cfg.visualization.threshold,
                output_file=output_file,
                cmap=cfg.visualization.colormap,
                vmax=1.0,
                engine=cfg.visualization.engine,
            )

            logger.info(f"Saved weights visualization to {output_file}")


if __name__ == "__main__":
    run_visualization()

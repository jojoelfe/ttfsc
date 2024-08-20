from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import mrcfile
import torch
import typer
from rich import print as rprint
from torch_fourier_shell_correlation import fsc

cli = typer.Typer(name="ttfsc", no_args_is_help=True, add_completion=False)


class Masking(str, Enum):
    none = "none"
    sphere = "sphere"


@cli.command(no_args_is_help=True)
def ttfsc_cli(
    map1: Annotated[Path, typer.Argument(show_default=False)],
    map2: Annotated[Path, typer.Argument(show_default=False)],
    pixel_spacing_angstroms: Annotated[
        Optional[float],
        typer.Option(
            "--pixel-spacing-angstroms", show_default=False, help="Pixel spacing in Å/px, taken from header if not set"
        ),
    ] = None,
    plot: Annotated[bool, typer.Option("--plot")] = True,
    plot_with_matplotlib: Annotated[bool, typer.Option("--plot-with-matplotlib")] = False,
    fsc_threshold: Annotated[float, typer.Option("--fsc-threshold", help="FSC threshold")] = 0.143,
    mask: Annotated[Masking, typer.Option("--mask")] = Masking.none,
    mask_radius: Annotated[float, typer.Option("--mask-radius")] = 100.0,
    mask_soft_edge_width: Annotated[int, typer.Option("--mask-soft-edge-width")] = 10,
) -> None:
    with mrcfile.open(map1) as f:
        map1_tensor = torch.tensor(f.data)
        if pixel_spacing_angstroms is None:
            pixel_spacing_angstroms = f.voxel_size.x
    with mrcfile.open(map2) as f:
        map2_tensor = torch.tensor(f.data)

    frequency_pixels = torch.fft.rfftfreq(map1_tensor.shape[0])
    resolution_angstroms = (1 / frequency_pixels) * pixel_spacing_angstroms

    fsc_values_unmasked = fsc(map1_tensor, map2_tensor)
    fsc_values_masked = None

    estimated_resolution_frequency_pixel = float(frequency_pixels[(fsc_values_unmasked < fsc_threshold).nonzero()[0] - 1])
    estimated_resolution_angstrom = float(resolution_angstroms[(fsc_values_unmasked < fsc_threshold).nonzero()[0] - 1])

    if mask == Masking.sphere:
        import numpy as np
        from ttmask.box_setup import box_setup
        from ttmask.soft_edge import add_soft_edge
        # Taken from https://github.com/teamtomo/ttmask/blob/main/src/ttmask/sphere.py

        # establish our coordinate system and empty mask
        coordinates_centered, mask_tensor = box_setup(map1_tensor.shape[0])

        # determine distances of each pixel to the center
        distance_to_center = np.linalg.norm(coordinates_centered, axis=-1)

        # set up criteria for which pixels are inside the sphere and modify values to 1.
        inside_sphere = distance_to_center < (mask_radius / pixel_spacing_angstroms)
        mask_tensor[inside_sphere] = 1

        # if requested, a soft edge is added to the mask
        mask_tensor = add_soft_edge(mask_tensor, mask_soft_edge_width)

        map1_tensor = map1_tensor * mask_tensor
        map2_tensor = map2_tensor * mask_tensor
        fsc_values_masked = fsc(map1_tensor, map2_tensor)

        estimated_resolution_frequency_pixel = float(frequency_pixels[(fsc_values_masked < fsc_threshold).nonzero()[0] - 1])
        estimated_resolution_angstrom = float(resolution_angstroms[(fsc_values_masked < fsc_threshold).nonzero()[0] - 1])

    rprint(f"Estimated resolution using {fsc_threshold} criterion: {estimated_resolution_angstrom:.2f} Å")

    if plot:
        from ._plotting import plot_matplotlib, plot_plottile

        if plot_with_matplotlib:
            plot_matplotlib(
                fsc_values_unmasked=fsc_values_unmasked,
                fsc_values_masked=fsc_values_masked,
                resolution_angstroms=resolution_angstroms,
                estimated_resolution_angstrom=estimated_resolution_angstrom,
                fsc_threshold=fsc_threshold,
            )
        else:
            plot_plottile(
                fsc_values=fsc_values_unmasked,
                fsc_values_masked=fsc_values_masked,
                frequency_pixels=frequency_pixels,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                estimated_resolution_frequency_pixel=estimated_resolution_frequency_pixel,
                fsc_threshold=fsc_threshold,
            )

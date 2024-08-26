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
    map1: Annotated[Path, typer.Argument()],
    map2: Annotated[Path, typer.Argument()],
    pixel_spacing_angstroms: Annotated[
        Optional[float],
        typer.Option(
            "--pixel-spacing-angstroms",
            show_default=False,
            help="Pixel spacing in Å/px, taken from header if not set",
            rich_help_panel="General options",
        ),
    ] = None,
    fsc_threshold: Annotated[
        float, typer.Option("--fsc-threshold", help="FSC threshold", rich_help_panel="General options")
    ] = 0.143,
    save_starfile: Annotated[
        Optional[Path],
        typer.Option("--save-starfile", help="Save FSC curves as a starfile", rich_help_panel="General options"),
    ] = None,
    plot: Annotated[bool, typer.Option("--plot", rich_help_panel="Plotting options")] = True,
    plot_with_matplotlib: Annotated[
        bool, typer.Option("--plot-with-matplotlib", rich_help_panel="Plotting options")
    ] = False,
    plot_matplotlib_style: Annotated[
        str, typer.Option("--plot-matplotlib-style", rich_help_panel="Plotting options")
    ] = "default",
    mask: Annotated[Masking, typer.Option("--mask", rich_help_panel="Masking options")] = Masking.none,
    mask_radius: Annotated[float, typer.Option("--mask-radius", rich_help_panel="Masking options")] = 100.0,
    mask_soft_edge_width: Annotated[int, typer.Option("--mask-soft-edge-width", rich_help_panel="Masking options")] = 10,
    correct_for_masking: Annotated[
        bool, typer.Option("--correct-for-masking", rich_help_panel="Masking correction options")
    ] = True,
    correct_from_resolution: Annotated[
        Optional[float], typer.Option("--correct-from_resolution", rich_help_panel="Masking correction options")
    ] = 10.0,
    correct_from_fraction_of_estimated_resolution: Annotated[
        float, typer.Option("--correct-from-fraction-of-estimated-resolution", rich_help_panel="Masking correction options")
    ] = 0.25,
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

        if correct_for_masking:
            from torch_grid_utils import fftfreq_grid

            map1_tensor_randomized = torch.fft.rfftn(map1_tensor)
            map2_tensor_randomized = torch.fft.rfftn(map2_tensor)
            frequency_grid = fftfreq_grid(
                image_shape=map1_tensor.shape,
                rfft=True,
                fftshift=False,
                norm=True,
                device=map1_tensor_randomized.device,
            )
            if correct_from_resolution is not None:
                to_correct = frequency_grid > (1 / correct_from_resolution) / pixel_spacing_angstroms
            else:
                to_correct = (
                    frequency_grid > correct_from_fraction_of_estimated_resolution * estimated_resolution_frequency_pixel
                )
            # Rotate phases at frequencies higher than 0.25
            random_phases1 = torch.rand(frequency_grid[to_correct].shape) * 2 * torch.pi
            random_phases1 = torch.complex(torch.cos(random_phases1), torch.sin(random_phases1))
            random_phases2 = torch.rand(frequency_grid[to_correct].shape) * 2 * torch.pi
            random_phases2 = torch.complex(torch.cos(random_phases2), torch.sin(random_phases2))

            map1_tensor_randomized[to_correct] *= random_phases1
            map2_tensor_randomized[to_correct] *= random_phases2

            map1_tensor_randomized = torch.fft.irfftn(map1_tensor_randomized)
            map2_tensor_randomized = torch.fft.irfftn(map2_tensor_randomized)

            map1_tensor_randomized *= mask_tensor
            map2_tensor_randomized *= mask_tensor
            fsc_values_masked_randomized = fsc(map1_tensor_randomized, map2_tensor_randomized)
        map1_tensor = map1_tensor * mask_tensor
        map2_tensor = map2_tensor * mask_tensor
        fsc_values_masked = fsc(map1_tensor, map2_tensor)
        if correct_for_masking:
            if correct_from_resolution is None:
                to_correct = (
                    frequency_pixels > correct_from_fraction_of_estimated_resolution * estimated_resolution_frequency_pixel
                )
            else:
                to_correct = frequency_pixels > (1 / correct_from_resolution) / pixel_spacing_angstroms
            fsc_values_corrected = fsc_values_masked.clone()
            fsc_values_corrected[to_correct] = (
                fsc_values_corrected[to_correct] - fsc_values_masked_randomized[to_correct]
            ) / (1.0 - fsc_values_masked_randomized[to_correct])

        estimated_resolution_frequency_pixel = float(frequency_pixels[(fsc_values_masked < fsc_threshold).nonzero()[0] - 1])
        estimated_resolution_angstrom = float(resolution_angstroms[(fsc_values_masked < fsc_threshold).nonzero()[0] - 1])

    rprint(f"Estimated resolution using {fsc_threshold} criterion: {estimated_resolution_angstrom:.2f} Å")

    if plot:
        from ._plotting import plot_matplotlib, plot_plottile

        if plot_with_matplotlib:
            plot_matplotlib(
                fsc_values_unmasked=fsc_values_unmasked,
                fsc_values_masked=fsc_values_masked,
                fsc_values_corrected=fsc_values_corrected,
                resolution_angstroms=resolution_angstroms,
                estimated_resolution_angstrom=estimated_resolution_angstrom,
                fsc_threshold=fsc_threshold,
                plot_matplotlib_style=plot_matplotlib_style,
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

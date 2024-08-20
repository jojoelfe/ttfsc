from pathlib import Path
from typing import Annotated, Optional

import mrcfile
import torch
import typer
from rich import print as rprint
from torch_fourier_shell_correlation import fsc

cli = typer.Typer(name="ttfsc", no_args_is_help=True, add_completion=False)


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
    fsc_threshold: Annotated[float, typer.Option("--fsc-threshold", show_default=False, help="FSC threshold")] = 0.143,
    # mask: Annotated[]
) -> None:
    with mrcfile.open(map1) as f:
        map1_tensor = torch.tensor(f.data)
        if pixel_spacing_angstroms is None:
            pixel_spacing_angstroms = f.voxel_size.x
    with mrcfile.open(map2) as f:
        map2_tensor = torch.tensor(f.data)

    frequency_pixels = torch.fft.rfftfreq(map1_tensor.shape[0])
    resolution_angstroms = (1 / frequency_pixels) * pixel_spacing_angstroms
    fsc_values = fsc(map1_tensor, map2_tensor)

    estimated_resolution_frequency_pixel = float(frequency_pixels[(fsc_values < fsc_threshold).nonzero()[0] - 1])
    estimated_resolution_angstrom = float(resolution_angstroms[(fsc_values < fsc_threshold).nonzero()[0] - 1])
    rprint(f"Estimated resolution using {fsc_threshold} criterion: {estimated_resolution_angstrom:.2f} Å")

    if plot:
        from ._plotting import plot_matplotlib, plot_plottile

        if plot_with_matplotlib:
            plot_matplotlib(
                fsc_values=fsc_values,
                resolution_angstroms=resolution_angstroms,
                estimated_resolution_angstrom=estimated_resolution_angstrom,
                fsc_threshold=fsc_threshold,
            )
        else:
            plot_plottile(
                fsc_values=fsc_values,
                frequency_pixels=frequency_pixels,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                estimated_resolution_frequency_pixel=estimated_resolution_frequency_pixel,
                fsc_threshold=fsc_threshold,
            )

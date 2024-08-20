from pathlib import Path
from typing import Optional, Annotated

import mrcfile
import typer
import torch
from torch_fourier_shell_correlation import fsc


cli = typer.Typer(name='ttctf', no_args_is_help=True, add_completion=False)

@cli.command(no_args_is_help=True)
def ttfsc_cli(
    map1: Annotated[Path, typer.Argument(show_default=False)],
    map2: Annotated[Path, typer.Argument(show_default=False)],
    pixel_spacing_angstroms: Annotated[Optional[float], typer.Option('--pixel-spacing-angstroms', show_default=False, help="Pixel spacing in Å/px, taken from header if not set")] = None,
    plot_with_matplotlib: Annotated[bool, typer.Option('--plot-with-matplotlib')] = False
):
    with mrcfile.open(map1) as f:
        map1_tensor = torch.tensor(f.data)
        if pixel_spacing_angstroms is None:
            pixel_spacing_angstroms = f.voxel_size.x
    with mrcfile.open(map2) as f:
        map2_tensor = torch.tensor(f.data)
    
    
    bin_centers = torch.fft.rfftfreq(map1_tensor.shape[0])
    resolution_angstrom = 1 / (bin_centers * pixel_spacing_angstroms)
    fsc_values = fsc(map1_tensor, map2_tensor)

    if plot_with_matplotlib:
        from matplotlib import pyplot as plt
        plt.plot(resolution_angstrom, fsc_values)
        plt.xlabel('Resolution (Å)')
        plt.xscale('log')
        plt.xlim(resolution_angstrom[1], resolution_angstrom[-2])

        plt.show()
        typer.Exit()
    
    import plotille

    fig = plotille.Figure()
    fig.width = 60
    fig.height = 20
    fig.set_x_limits(float(bin_centers[1]), float(bin_centers[-1]))
    fig.set_y_limits(0, 1)
    def resolution_callback(x,_):
        return '{:.2f}'.format(1 / (x * pixel_spacing_angstroms))
    fig.x_ticks_fkt = resolution_callback
    fig.plot(bin_centers[1:].numpy(), fsc_values[1:].numpy(),label='FSC')
    print(fig.show(legend=True))
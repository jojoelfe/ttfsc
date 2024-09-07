from pathlib import Path
from typing import Annotated, Optional

import mrcfile
import torch
import typer
from rich import print as rprint
from torch_fourier_shell_correlation import fsc

from ._masking import Masking

cli = typer.Typer(name="ttfsc", no_args_is_help=True, add_completion=False)


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
    mask_radius_angstroms: Annotated[
        float, typer.Option("--mask-radius-angstroms", rich_help_panel="Masking options")
    ] = 100.0,
    mask_soft_edge_width_pixels: Annotated[
        int, typer.Option("--mask-soft-edge-width-pixels", rich_help_panel="Masking options")
    ] = 10,
    correct_for_masking: Annotated[
        bool, typer.Option("--correct-for-masking", rich_help_panel="Masking correction options")
    ] = True,
    correct_from_resolution: Annotated[
        Optional[float], typer.Option("--correct-from_resolution", rich_help_panel="Masking correction options")
    ] = None,
    correct_from_fraction_of_estimated_resolution: Annotated[
        float, typer.Option("--correct-from-fraction-of-estimated-resolution", rich_help_panel="Masking correction options")
    ] = 0.5,
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

    estimated_resolution_frequency_pixel = float(frequency_pixels[(fsc_values_unmasked < fsc_threshold).nonzero()[0] - 1])
    estimated_resolution_angstrom = float(resolution_angstroms[(fsc_values_unmasked < fsc_threshold).nonzero()[0] - 1])

    rprint(f"Estimated resolution using {fsc_threshold} criterion in unmasked map: {estimated_resolution_angstrom:.2f} Å")

    fsc_values_masked = None
    if mask != Masking.none:
        from ._masking import calculate_masked_fsc

        (estimated_resolution_angstrom, estimated_resolution_frequency_pixel, fsc_values_masked, mask_tensor) = (
            calculate_masked_fsc(
                map1_tensor,
                map2_tensor,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                fsc_threshold=fsc_threshold,
                mask=mask,
                mask_radius_angstroms=mask_radius_angstroms,
                mask_soft_edge_width_pixels=mask_soft_edge_width_pixels,
            )
        )
        rprint(f"Estimated resolution using {fsc_threshold} criterion in masked map: {estimated_resolution_angstrom:.2f} Å")
        if correct_for_masking:
            from ._masking import calculate_noise_injected_fsc

            (
                estimated_resolution_angstrom,
                estimated_resolution_frequency_pixel,
                correction_from_resolution_angstrom,
                fsc_values_corrected,
                fsc_values_randomized,
            ) = calculate_noise_injected_fsc(
                map1_tensor,
                map2_tensor,
                mask_tensor=mask_tensor,
                fsc_values_masked=fsc_values_masked,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                fsc_threshold=fsc_threshold,
                estimated_resolution_frequency_pixel=estimated_resolution_frequency_pixel,
                correct_from_resolution=correct_from_resolution,
                correct_from_fraction_of_estimated_resolution=correct_from_fraction_of_estimated_resolution,
            )
            rprint(
                f"Estimated resolution using {fsc_threshold} "
                f"criterion with correction after {correction_from_resolution_angstrom:.2f} Å: "
                f"{estimated_resolution_angstrom:.2f} Å"
            )
    if save_starfile:
        import pandas as pd
        import starfile
        from numpy import nan

        from ._starfile_schema import RelionDataGeneral, RelionFSCData

        data_general = RelionDataGeneral(
            rlnFinalResolution=estimated_resolution_angstrom,
            rlnUnfilteredMapHalf1=map1.name,
            rlnUnfilteredMapHalf2=map2.name,
        )
        if mask != Masking.none:
            data_general.rlnParticleBoxFractionSolventMask = mask_tensor.sum().item() / mask_tensor.numel()
        if correct_for_masking:
            data_general.rlnRandomiseFrom = correction_from_resolution_angstrom

        fsc_data = []
        for i, (f, r) in enumerate(zip(fsc_values_unmasked, resolution_angstroms)):
            fsc_data.append(
                RelionFSCData(
                    rlnSpectralIndex=i,
                    rlnResolution=r,
                    rlnAngstromResolution=r,
                    rlnFourierShellCorrelationCorrected=fsc_values_corrected[i] if correct_for_masking else nan,
                    rlnFourierShellCorrelationUnmaskedMaps=f,
                    rlnFourierShellCorrelationMaskedMaps=fsc_values_masked[i] if fsc_values_masked is not None else nan,
                    rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps=fsc_values_randomized[i]
                    if correct_for_masking
                    else nan,
                    rlnFourierShellCorrelationParticleMaskFraction=f
                    / data_general.rlnParticleBoxFractionSolventMask
                    / (1.0 + (1.0 / data_general.rlnParticleBoxFractionSolventMask - 1.0) * f.abs())
                    if mask != Masking.none
                    else nan,
                )
            )
        starfile.write(
            {"general": data_general.__dict__, "fsc": pd.DataFrame([f.__dict__ for f in fsc_data])}, save_starfile
        )

    if plot:
        from ._plotting import plot_matplotlib, plot_plottile

        if plot_with_matplotlib:
            plot_matplotlib(
                fsc_values_unmasked=fsc_values_unmasked,
                fsc_values_masked=fsc_values_masked,
                fsc_values_corrected=fsc_values_corrected,
                frequency_pixels=frequency_pixels,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                estimated_resolution_frequency_pixel=estimated_resolution_frequency_pixel,
                fsc_threshold=fsc_threshold,
                plot_matplotlib_style=plot_matplotlib_style,
            )
        else:
            plot_plottile(
                fsc_values=fsc_values_unmasked,
                fsc_values_masked=fsc_values_masked,
                fsc_values_corrected=fsc_values_corrected,
                frequency_pixels=frequency_pixels,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                estimated_resolution_frequency_pixel=estimated_resolution_frequency_pixel,
                fsc_threshold=fsc_threshold,
            )

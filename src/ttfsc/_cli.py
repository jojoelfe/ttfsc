from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint

from ._masking import Masking
from .ttfsc import ttfsc

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
    result = ttfsc(
        map1=map1,
        map2=map2,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        fsc_threshold=fsc_threshold,
        mask=mask,
        mask_radius_angstroms=mask_radius_angstroms,
        mask_soft_edge_width_pixels=mask_soft_edge_width_pixels,
        correct_for_masking=correct_for_masking,
        correct_from_resolution=correct_from_resolution,
        correct_from_fraction_of_estimated_resolution=correct_from_fraction_of_estimated_resolution,
    )

    rprint(
        f"Estimated resolution using {fsc_threshold} criterion in unmasked map: "
        f"{result.estimated_resolution_angstrom_unmasked:.2f} Å"
    )
    if result.estimated_resolution_angstrom_masked is not None:
        rprint(
            f"Estimated resolution using {fsc_threshold} "
            f"criterion in masked map: {result.estimated_resolution_angstrom_masked:.2f} Å"
        )
    if result.estimated_resolution_angstrom_corrected is not None:
        print(
            f"Estimated resolution using {fsc_threshold} "
            f"criterion with correction after {result.correction_from_resolution_angstrom:.2f} Å: "
            f"{result.estimated_resolution_angstrom_corrected:.2f} Å"
        )

    if save_starfile:
        import pandas as pd
        import starfile
        from numpy import nan

        from ._data_models import RelionDataGeneral, RelionFSCData

        data_general = RelionDataGeneral(
            rlnFinalResolution=result.estimated_resolution_angstrom,
            rlnUnfilteredMapHalf1=str(result.map1),
            rlnUnfilteredMapHalf2=str(result.map2),
        )
        if result.mask_tensor is not None:
            data_general.rlnParticleBoxFractionSolventMask = result.mask_tensor.sum().item() / result.mask_tensor.numel()
        if correct_for_masking:
            if result.correction_from_resolution_angstrom is None:
                raise ValueError("Phase randomization cutoff has not been calculated")
            data_general.rlnRandomiseFrom = result.correction_from_resolution_angstrom

        fsc_data = []
        for i in range(result.num_shells):
            fsc_data.append(
                RelionFSCData(
                    rlnSpectralIndex=i,
                    rlnResolution=result.frequency_pixels[i],
                    rlnAngstromResolution=result.resolution_angstroms[i],
                    rlnFourierShellCorrelationCorrected=result.fsc_values_corrected[i]
                    if result.fsc_values_corrected is not None
                    else nan,
                    rlnFourierShellCorrelationUnmaskedMaps=result.fsc_values_unmasked[i],
                    rlnFourierShellCorrelationMaskedMaps=result.fsc_values_masked[i]
                    if result.fsc_values_masked is not None
                    else nan,
                    rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps=result.fsc_values_randomized[i]
                    if result.fsc_values_randomized is not None
                    else nan,
                    rlnFourierShellCorrelationParticleMaskFraction=result.fsc_values_unmasked[i]
                    / data_general.rlnParticleBoxFractionSolventMask
                    / (
                        1.0
                        + (1.0 / data_general.rlnParticleBoxFractionSolventMask - 1.0) * result.fsc_values_unmasked[i].abs()
                    )
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
            plot_matplotlib(result, plot_matplotlib_style=plot_matplotlib_style)
        else:
            plot_plottile(result)

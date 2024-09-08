"""
Provides functionality for Fourier Shell Correlation (FSC) analysis.

The main function in this module is `ttfsc`, which calculates the FSC between two
3D maps and returns a `TTFSCResult` object containing the results of the analysis.

Example:
    ttfsc("map1.mrc","map2.mrc")
"""

from pathlib import Path
from typing import Optional

import mrcfile
import torch
from torch_fourier_shell_correlation import fsc

from ._data_models import TTFSCResult
from ._masking import Masking


def ttfsc(
    map1: Path,
    map2: Path,
    pixel_spacing_angstroms: Optional[float] = None,
    fsc_threshold: float = 0.143,
    mask: Masking = Masking.none,
    mask_radius_angstroms: float = 100.0,
    mask_soft_edge_width_pixels: int = 10,
    correct_for_masking: bool = True,
    correct_from_resolution: Optional[float] = None,
    correct_from_fraction_of_estimated_resolution: float = 0.5,
) -> TTFSCResult:
    """
    Perform Fourier Shell Correlation (FSC) analysis between two maps.

    Args:
        map1 (Path): Path to the first map file.
        map2 (Path): Path to the second map file.
        pixel_spacing_angstroms (Optional[float]): Pixel spacing in Å/px. If not provided, it will be taken from the header.
        fsc_threshold (float): FSC threshold value. Default is 0.143.
        mask (Masking): Masking option to use. Default is Masking.none.
        mask_radius_angstroms (float): Radius of the mask in Å. Default is 100.0.
        mask_soft_edge_width_pixels (int): Width of the soft edge of the mask in pixels. Default is 10.
        correct_for_masking (bool): Whether to correct for masking effects. Default is True.
        correct_from_resolution (Optional[float]): Resolution from which to start correction.
                                                   Default is None.
        correct_from_fraction_of_estimated_resolution (float): Fraction of the estimated resolution
                                                               from which to start correction. Default is 0.5.

    Returns
    -------
        TTFSCResult: The result of the FSC analysis, including FSC curves and resolution estimates.

    Example:
        result = ttfsc(
            map1=Path("map1.mrc"),
            map2=Path("map2.mrc"),
            pixel_spacing_angstroms=1.0,
            fsc_threshold=0.143,
            mask=Masking.soft,
            mask_radius_angstroms=150.0,
            mask_soft_edge_width_pixels=5,
            correct_for_masking=True,
            correct_from_resolution=3.0,
            correct_from_fraction_of_estimated_resolution=0.5
        )
    """
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
    result = TTFSCResult(
        map1=map1,
        map1_tensor=map1_tensor,
        map2=map2,
        map2_tensor=map2_tensor,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        fsc_threshold=fsc_threshold,
        num_shells=len(frequency_pixels),
        estimated_resolution_angstrom=estimated_resolution_angstrom,
        estimated_resolution_angstrom_unmasked=estimated_resolution_angstrom,
        estimated_resolution_frequency_pixel=estimated_resolution_frequency_pixel,
        frequency_pixels=frequency_pixels,
        resolution_angstroms=resolution_angstroms,
        fsc_values_unmasked=fsc_values_unmasked,
    )
    if mask != Masking.none:
        from ._masking import calculate_masked_fsc

        result.mask = mask
        result.mask_radius_angstroms = mask_radius_angstroms
        result.mask_soft_edge_width_pixels = mask_soft_edge_width_pixels
        calculate_masked_fsc(result)
        if correct_for_masking:
            from ._masking import calculate_noise_injected_fsc

            result.correction_from_resolution_angstrom = correct_from_resolution
            result.correct_from_fraction_of_estimated_resolution = correct_from_fraction_of_estimated_resolution
            calculate_noise_injected_fsc(result)
    return result

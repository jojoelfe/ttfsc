from enum import Enum
from typing import Optional

import torch
from torch_fourier_shell_correlation import fsc


class Masking(str, Enum):
    none = "none"
    sphere = "sphere"


def calculate_noise_injected_fsc(
    map1_tensor: torch.tensor,
    map2_tensor: torch.tensor,
    mask_tensor: torch.tensor,
    fsc_values_masked: torch.tensor,
    pixel_spacing_angstroms: float,
    fsc_threshold: float,
    estimated_resolution_frequency_pixel: float,
    correct_from_resolution: Optional[float] = None,
    correct_from_fraction_of_estimated_resolution: float = 0.5,
):
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
        to_correct = frequency_grid > correct_from_fraction_of_estimated_resolution * estimated_resolution_frequency_pixel
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

    frequency_pixels = torch.fft.rfftfreq(map1_tensor.shape[0])
    resolution_angstroms = (1 / frequency_pixels) * pixel_spacing_angstroms

    if correct_from_resolution is None:
        to_correct = frequency_pixels > correct_from_fraction_of_estimated_resolution * estimated_resolution_frequency_pixel
        correct_from_resolution = pixel_spacing_angstroms / (
            correct_from_fraction_of_estimated_resolution * estimated_resolution_frequency_pixel
        )
    else:
        to_correct = frequency_pixels > (1 / correct_from_resolution) / pixel_spacing_angstroms
    fsc_values_corrected = fsc_values_masked.clone()
    fsc_values_corrected[to_correct] = (fsc_values_corrected[to_correct] - fsc_values_masked_randomized[to_correct]) / (
        1.0 - fsc_values_masked_randomized[to_correct]
    )

    estimated_resolution_frequency_pixel = float(frequency_pixels[(fsc_values_corrected < fsc_threshold).nonzero()[0] - 1])
    estimated_resolution_angstrom = float(resolution_angstroms[(fsc_values_corrected < fsc_threshold).nonzero()[0] - 1])

    return (
        estimated_resolution_angstrom,
        estimated_resolution_frequency_pixel,
        correct_from_resolution,
        fsc_values_corrected,
    )


def calculate_masked_fsc(
    map1_tensor: torch.tensor,
    map2_tensor: torch.tensor,
    pixel_spacing_angstroms: float,
    fsc_threshold: float,
    mask: Masking,
    mask_radius_angstroms: float = 100.0,
    mask_soft_edge_width_pixels: int = 5,
) -> tuple[float, float, torch.tensor, torch.tensor]:
    if mask == Masking.none:
        raise ValueError("Must choose a mask type")
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
        inside_sphere = distance_to_center < (mask_radius_angstroms / pixel_spacing_angstroms)
        mask_tensor[inside_sphere] = 1

        # if requested, a soft edge is added to the mask
        mask_tensor = add_soft_edge(mask_tensor, mask_soft_edge_width_pixels)

        map1_tensor_masked = map1_tensor * mask_tensor
        map2_tensor_masked = map2_tensor * mask_tensor
        fsc_values_masked = fsc(map1_tensor_masked, map2_tensor_masked)

        frequency_pixels = torch.fft.rfftfreq(map1_tensor.shape[0])
        resolution_angstroms = (1 / frequency_pixels) * pixel_spacing_angstroms

        estimated_resolution_frequency_pixel = float(frequency_pixels[(fsc_values_masked < fsc_threshold).nonzero()[0] - 1])
        estimated_resolution_angstrom = float(resolution_angstroms[(fsc_values_masked < fsc_threshold).nonzero()[0] - 1])

        return (estimated_resolution_angstrom, estimated_resolution_frequency_pixel, fsc_values_masked, mask_tensor)

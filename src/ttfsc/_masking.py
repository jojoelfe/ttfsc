import torch
from torch_fourier_shell_correlation import fsc

from ._data_models import Masking, TTFSCResult


def calculate_noise_injected_fsc(result: TTFSCResult) -> None:
    from torch_grid_utils import fftfreq_grid

    map1_tensor_randomized = torch.fft.rfftn(result.map1_tensor)
    map2_tensor_randomized = torch.fft.rfftn(result.map2_tensor)
    frequency_grid = fftfreq_grid(
        image_shape=result.map1_tensor.shape,
        rfft=True,
        fftshift=False,
        norm=True,
        device=map1_tensor_randomized.device,
    )
    if result.correction_from_resolution_angstrom is not None:
        to_correct = frequency_grid > (1 / result.correction_from_resolution_angstrom) / result.pixel_spacing_angstroms
    else:
        to_correct = (
            frequency_grid
            > result.correct_from_fraction_of_estimated_resolution * result.estimated_resolution_frequency_pixel
        )

    random_phases1 = torch.rand(frequency_grid[to_correct].shape) * 2 * torch.pi
    random_phases1 = torch.complex(torch.cos(random_phases1), torch.sin(random_phases1))
    random_phases2 = torch.rand(frequency_grid[to_correct].shape) * 2 * torch.pi
    random_phases2 = torch.complex(torch.cos(random_phases2), torch.sin(random_phases2))

    map1_tensor_randomized[to_correct] *= random_phases1
    map2_tensor_randomized[to_correct] *= random_phases2

    map1_tensor_randomized = torch.fft.irfftn(map1_tensor_randomized)
    map2_tensor_randomized = torch.fft.irfftn(map2_tensor_randomized)

    map1_tensor_randomized *= result.mask_tensor
    map2_tensor_randomized *= result.mask_tensor
    result.fsc_values_masked_randomized = fsc(map1_tensor_randomized, map2_tensor_randomized)

    if result.correction_from_resolution_angstrom is None:
        to_correct = (
            result.frequency_pixels
            > result.correct_from_fraction_of_estimated_resolution * result.estimated_resolution_frequency_pixel
        )
        result.correction_from_resolution_angstrom = result.pixel_spacing_angstroms / (
            result.correct_from_fraction_of_estimated_resolution * result.estimated_resolution_frequency_pixel
        )
    else:
        to_correct = (
            result.frequency_pixels > (1 / result.correction_from_resolution_angstrom) / result.pixel_spacing_angstroms
        )
    if result.fsc_values_masked is None:
        raise ValueError("Must calculate masked FSC before correcting for masking")
    result.fsc_values_corrected = result.fsc_values_masked.clone()
    result.fsc_values_corrected[to_correct] = (
        result.fsc_values_corrected[to_correct] - result.fsc_values_masked_randomized[to_correct]
    ) / (1.0 - result.fsc_values_masked_randomized[to_correct])

    result.estimated_resolution_frequency_pixel = float(
        result.frequency_pixels[(result.fsc_values_corrected < result.fsc_threshold).nonzero()[0] - 1]
    )
    result.estimated_resolution_angstrom = float(
        result.resolution_angstroms[(result.fsc_values_corrected < result.fsc_threshold).nonzero()[0] - 1]
    )
    result.estimated_resolution_angstrom_corrected = result.estimated_resolution_angstrom


def calculate_masked_fsc(result: TTFSCResult) -> None:
    if result.mask == Masking.none:
        raise ValueError("Must choose a mask type")
    if result.mask == Masking.sphere:
        import numpy as np
        from ttmask.box_setup import box_setup
        from ttmask.soft_edge import add_soft_edge
        # Taken from https://github.com/teamtomo/ttmask/blob/main/src/ttmask/sphere.py

        # establish our coordinate system and empty mask
        coordinates_centered, mask_tensor = box_setup(result.map1_tensor.shape[0])

        # determine distances of each pixel to the center
        distance_to_center = np.linalg.norm(coordinates_centered, axis=-1)

        # set up criteria for which pixels are inside the sphere and modify values to 1.
        inside_sphere = distance_to_center < (result.mask_radius_angstroms / result.pixel_spacing_angstroms)
        mask_tensor[inside_sphere] = 1

        # if requested, a soft edge is added to the mask
        result.mask_tensor = torch.tensor(add_soft_edge(mask_tensor, result.mask_soft_edge_width_pixels))

        map1_tensor_masked = result.map1_tensor * result.mask_tensor
        map2_tensor_masked = result.map2_tensor * result.mask_tensor
        result.fsc_values_masked = fsc(map1_tensor_masked, map2_tensor_masked)

        result.estimated_resolution_frequency_pixel = float(
            result.frequency_pixels[(result.fsc_values_masked < result.fsc_threshold).nonzero()[0] - 1]
        )
        result.estimated_resolution_angstrom = float(
            result.resolution_angstroms[(result.fsc_values_masked < result.fsc_threshold).nonzero()[0] - 1]
        )
        result.estimated_resolution_angstrom_masked = result.estimated_resolution_angstrom

        return
    raise NotImplementedError("Only sphere masking is implemented")

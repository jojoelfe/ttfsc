from enum import Enum
from pathlib import Path
from typing import List, Optional

from numpy import nan
from pydantic import BaseModel
from torch import Tensor


class Masking(str, Enum):
    none = "none"
    sphere = "sphere"


class TTFSCResult(BaseModel):
    map1: Path
    map1_tensor: Tensor
    map2: Path
    map2_tensor: Tensor
    pixel_spacing_angstroms: float
    fsc_threshold: float
    mask: str = Masking.none
    mask_filename: Optional[Path] = None
    mask_tensor: Optional[Tensor] = None
    mask_radius_angstroms: float = 50.0
    mask_soft_edge_width_pixels: int = 10
    num_shells: int
    estimated_resolution_angstrom: float
    estimated_resolution_angstrom_unmasked: float
    estimated_resolution_angstrom_masked: Optional[float] = None
    estimated_resolution_angstrom_corrected: Optional[float] = None
    estimated_resolution_frequency_pixel: float
    frequency_pixels: Tensor
    resolution_angstroms: Tensor
    fsc_values_unmasked: Tensor
    fsc_values_masked: Optional[Tensor] = None
    fsc_values_corrected: Optional[Tensor] = None
    fsc_values_masked_randomized: Optional[Tensor] = None
    fsc_values_randomized: Optional[Tensor] = None
    correction_from_resolution_angstrom: Optional[float] = None
    correct_from_fraction_of_estimated_resolution: float = 0.5

    model_config: dict = {"arbitrary_types_allowed": True}


class RelionDataGeneral(BaseModel):
    rlnFinalResolution: float
    rlnUnfilteredMapHalf1: str
    rlnUnfilteredMapHalf2: str
    rlnParticleBoxFractionSolventMask: float = nan
    rlnRandomiseFrom: float = nan
    rlnMaskName: str = ""


class RelionFSCData(BaseModel):
    rlnSpectralIndex: int
    rlnResolution: float
    rlnAngstromResolution: float
    rlnFourierShellCorrelationCorrected: float
    rlnFourierShellCorrelationParticleMaskFraction: float
    rlnFourierShellCorrelationUnmaskedMaps: float
    rlnFourierShellCorrelationMaskedMaps: float
    rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps: float


class RelionStarfile(BaseModel):
    data_general: RelionDataGeneral
    fsc_data: List[RelionFSCData]

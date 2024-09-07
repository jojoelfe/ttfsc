from typing import List

from numpy import nan
from pydantic import BaseModel


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

"""Settings for the OpenMM MM/GBSA Protocol.

The Protocol settings are a superset of
:class:`openfe.protocols.openmm_md.PlainMDProtocolSettings`, so that the
PlainMD setup and simulation units can be reused unchanged.
"""

from typing import Annotated, Literal, Optional, TypeAlias

from gufe.settings import SettingsBaseModel
from gufe.settings.typing import (
    GufeQuantity,
    NanometerQuantity,
    NanosecondQuantity,
    specify_quantity_units,
)
from openfe.protocols.openmm_md.plain_md_settings import PlainMDProtocolSettings
from openff.units import unit
from pydantic import ConfigDict, field_validator

MolarQuantity: TypeAlias = Annotated[GufeQuantity, specify_quantity_units("molar")]
GBSurfaceTensionQuantity: TypeAlias = Annotated[
    GufeQuantity, specify_quantity_units("kilocalorie_per_mole / angstrom**2")
]


class MMGBSASettings(SettingsBaseModel):
    """
    Settings controlling how frames are re-scored with MM/GBSA.

    The polar solvation term is computed with one of the Amber-style
    generalized Born models available in
    :mod:`openmm.app.internal.customgbforces`, using the radii set
    conventionally paired with that model. The non-polar solvation term is
    computed using the ACE approximation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gb_model: Literal["HCT", "OBC1", "OBC2", "GBn", "GBn2"] = "OBC2"
    """
    The generalized Born model to use. These correspond to Amber's
    ``igb=1`` (HCT, mbondi radii), ``igb=2`` (OBC1, mbondi2 radii),
    ``igb=5`` (OBC2, mbondi2 radii), ``igb=7`` (GBn, bondi radii)
    and ``igb=8`` (GBn2, mbondi3 radii). Default 'OBC2'.
    """

    solute_dielectric: float = 1.0
    """The dielectric constant of the solute. Default 1.0."""

    solvent_dielectric: float = 78.5
    """The dielectric constant of the implicit solvent. Default 78.5."""

    salt_concentration: MolarQuantity = 0.15 * unit.molar
    """
    The implicit salt concentration, used to compute the Debye-Huckel
    screening parameter. Default 0.15 molar.
    """

    surface_tension: GBSurfaceTensionQuantity = 0.0054 * unit.kilocalorie_per_mole / unit.angstrom**2
    """
    The surface tension used for the ACE non-polar solvation term.
    Default 0.0054 kcal/mol/A^2 (equivalent to the OpenMM default).
    """

    probe_radius: NanometerQuantity = 0.14 * unit.nanometer
    """The solvent probe radius for the ACE non-polar term. Default 0.14 nm."""

    n_waters: int = 0
    """
    The number of explicit waters closest to the ligand to retain in
    each frame and treat as part of the receptor (the MM/GBSA-NWAT
    approach). Default 0 (standard MM/GBSA).

    Note
    ----
    If this is greater than zero, ``output_settings.output_indices`` must
    select the solvent waters so that they are written to the trajectory.
    """

    water_selection: Literal["heavy", "all"] = "heavy"
    """
    Which ligand atoms to use when finding the closest waters for
    MM/GBSA-NWAT; either only the ligand heavy atoms ('heavy') or all
    ligand atoms ('all'). Distances are measured to water oxygens.
    Default 'heavy'.
    """

    analysis_start_time: Optional[NanosecondQuantity] = None
    """
    Production time after which frames are used for analysis. If ``None``,
    all production frames are used. Default ``None``.
    """

    frame_stride: int = 1
    """Analyze every ``frame_stride``-th frame. Default 1."""

    @field_validator("solute_dielectric", "solvent_dielectric")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"Dielectric constants must be positive, got {v}."
            raise ValueError(errmsg)
        return v

    @field_validator("salt_concentration", "surface_tension", "probe_radius")
    def must_be_non_negative(cls, v):
        if v.m < 0:
            errmsg = f"Value must be non-negative, got {v}."
            raise ValueError(errmsg)
        return v

    @field_validator("n_waters")
    def n_waters_non_negative(cls, v):
        if v < 0:
            errmsg = f"n_waters must be non-negative, got {v}."
            raise ValueError(errmsg)
        return v

    @field_validator("frame_stride")
    def stride_positive(cls, v):
        if v <= 0:
            errmsg = f"frame_stride must be positive, got {v}."
            raise ValueError(errmsg)
        return v

    @field_validator("analysis_start_time")
    def start_time_non_negative(cls, v):
        if v is not None and v.m < 0:
            errmsg = f"analysis_start_time must be non-negative, got {v}."
            raise ValueError(errmsg)
        return v


class MMGBSAProtocolSettings(PlainMDProtocolSettings):
    """
    Settings for the MM/GBSA Protocol.

    Extends :class:`openfe.protocols.openmm_md.PlainMDProtocolSettings`
    with :class:`MMGBSASettings`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mmgbsa_settings: MMGBSASettings
    """Settings for the MM/GBSA re-scoring of the trajectory."""

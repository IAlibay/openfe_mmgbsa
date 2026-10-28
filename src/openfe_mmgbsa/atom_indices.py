"""Utilities for identifying the receptor, ligand and water atoms of a System."""

import json
import pathlib
from dataclasses import asdict, dataclass

import mdtraj
import numpy as np
import openmm
from gufe import ProteinComponent, SmallMoleculeComponent, SolvatedPDBComponent
from openmm import app

_WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TIP", "SOL", "H2O", "T3P", "T4P", "T5P", "SPC"}

# Monatomic ions of a SolvatedPDBComponent which are considered to be part
# of the solvent rather than the receptor
_SOLVENT_ION_ELEMENTS = {"Li", "Na", "K", "Rb", "Cs", "F", "Cl", "Br", "I"}


@dataclass
class MMGBSAAtomIndices:
    """
    The atom indices, in the full explicit solvent System, of each part of
    the MM/GBSA calculation.
    """

    receptor: list[int]
    """Receptor atoms."""

    ligand: list[int]
    """Ligand atoms."""

    waters: list[list[int]]
    """Water molecules, each defined as the indices of [O, H, H]."""

    trajectory: list[int]
    """The atoms written to the production trajectory, in order."""

    def to_json(self, filename: pathlib.Path) -> None:
        with open(filename, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_json(cls, filename: pathlib.Path) -> "MMGBSAAtomIndices":
        with open(filename) as f:
            return cls(**json.load(f))


def _is_water(residue: app.Residue) -> bool:
    if residue.name.upper() in _WATER_RESNAMES:
        return True
    elements = sorted(a.element.symbol if a.element is not None else "" for a in residue.atoms())
    return elements == ["H", "H", "O"]


def _water_atoms(residue: app.Residue) -> list[int]:
    """
    Get the water atom indices as [O, H, H], raising an error for
    anything other than a 3-site water.
    """
    atoms = list(residue.atoms())
    oxygens = [a.index for a in atoms if a.element is not None and a.element.symbol == "O"]
    hydrogens = [a.index for a in atoms if a.element is not None and a.element.symbol == "H"]
    if len(atoms) != 3 or len(oxygens) != 1 or len(hydrogens) != 2:
        errmsg = (
            f"Water residue {residue} is not a 3-site water model, only 3-site "
            "water models are currently supported."
        )
        raise ValueError(errmsg)
    return oxygens + hydrogens


def get_mmgbsa_atom_indices(
    topology: app.Topology,
    protein_component: ProteinComponent,
    ligand_component: SmallMoleculeComponent,
    output_selection: str,
) -> MMGBSAAtomIndices:
    """
    Identify the receptor, ligand and water atoms of a System created by
    the PlainMD setup unit.

    The PlainMD setup creates the System by adding, in order, the protein
    component, the small molecule(s) and then any solvent. Within the protein
    component, water residues are treated as solvent (and are candidates for
    MM/GBSA-NWAT) and everything else is treated as receptor, except for the
    monatomic solvent ions (e.g. Na+, Cl-) of a ``SolvatedPDBComponent``.

    Parameters
    ----------
    topology : openmm.app.Topology
      The Topology of the full explicit solvent System.
    protein_component : ProteinComponent
      The protein component of the ChemicalSystem.
    ligand_component : SmallMoleculeComponent
      The ligand.
    output_selection : str
      The MDTraj selection string used to choose which atoms are written
      to the production trajectory.

    Returns
    -------
    MMGBSAAtomIndices
      The atom indices for each part of the MM/GBSA calculation.

    Raises
    ------
    ValueError
      If the ligand atoms cannot be found where expected, or if the
      trajectory does not contain all the receptor and ligand atoms.
    """
    atoms = list(topology.atoms())
    n_protein = protein_component.to_openmm_topology().getNumAtoms()
    ligand_offmol = ligand_component.to_openff()
    n_ligand = ligand_offmol.n_atoms

    # Check that the ligand is where we expect
    ligand = list(range(n_protein, n_protein + n_ligand))
    expected_elements = [a.symbol for a in ligand_offmol.atoms]
    found_elements = [
        atoms[i].element.symbol if (i < len(atoms) and atoms[i].element is not None) else None
        for i in ligand
    ]
    if expected_elements != found_elements:
        errmsg = (
            "Could not locate the ligand atoms in the System topology. The ligand "
            "atoms are expected to directly follow the protein atoms."
        )
        raise ValueError(errmsg)
    ligand_residues = {atoms[i].residue for i in ligand}
    if any(a.index not in ligand for r in ligand_residues for a in r.atoms()):
        raise ValueError("The ligand residue(s) contain atoms which are not part of the ligand")

    solvated = isinstance(protein_component, SolvatedPDBComponent)

    receptor = []
    waters = []
    for residue in topology.residues():
        if residue in ligand_residues:
            continue
        res_atoms = list(residue.atoms())
        in_protein = res_atoms[0].index < n_protein
        if _is_water(residue):
            waters.append(_water_atoms(residue))
        elif in_protein:
            is_solvent_ion = (
                len(res_atoms) == 1
                and res_atoms[0].element is not None
                and res_atoms[0].element.symbol in _SOLVENT_ION_ELEMENTS
            )
            if not (solvated and is_solvent_ion):
                receptor.extend(a.index for a in res_atoms)

    trajectory = mdtraj.Topology.from_openmm(topology).select(output_selection)
    trajectory = [int(i) for i in trajectory]

    missing = set(receptor + ligand) - set(trajectory)
    if missing:
        errmsg = (
            f"The output_indices selection '{output_selection}' does not include "
            f"{len(missing)} receptor and/or ligand atoms. All receptor and ligand "
            "atoms must be written to the trajectory for MM/GBSA."
        )
        raise ValueError(errmsg)

    return MMGBSAAtomIndices(
        receptor=sorted(receptor),
        ligand=ligand,
        waters=waters,
        trajectory=trajectory,
    )


def check_waters_in_trajectory(indices: MMGBSAAtomIndices, output_selection: str) -> None:
    """
    Check that all the waters are written to the trajectory, as needed for
    MM/GBSA-NWAT.
    """
    traj = set(indices.trajectory)
    missing = [w for w in indices.waters if not set(w).issubset(traj)]
    if missing:
        errmsg = (
            f"The output_indices selection '{output_selection}' does not include "
            f"{len(missing)} water molecules. All waters must be written to the "
            "trajectory when using MM/GBSA-NWAT (n_waters > 0), e.g. by using "
            "output_indices='all'."
        )
        raise ValueError(errmsg)


def check_identical_water_parameters(system: openmm.System, waters: list[list[int]]) -> None:
    """
    Check that all the waters have identical nonbonded parameters, as
    required for MM/GBSA-NWAT where the closest waters are placed into
    fixed water "slots".

    Only the nonbonded parameters are checked; any bonded water terms are
    identical in the complex and receptor and therefore cancel.
    """
    nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if len(nonbonded) != 1:
        raise ValueError("Expected exactly one NonbondedForce")
    nb = nonbonded[0]

    def _params(water):
        values = []
        for idx in water:
            q, sig, eps = nb.getParticleParameters(idx)
            values.append(
                (
                    q.value_in_unit(openmm.unit.elementary_charge),
                    sig.value_in_unit(openmm.unit.nanometer),
                    eps.value_in_unit(openmm.unit.kilojoule_per_mole),
                )
            )
        return np.array(values)

    reference = _params(waters[0])
    for water in waters[1:]:
        if not np.allclose(_params(water), reference):
            errmsg = (
                "All waters must have identical parameters for MM/GBSA-NWAT, "
                f"water with atoms {water} differs from water with atoms {waters[0]}."
            )
            raise ValueError(errmsg)

"""Utilities for building the generalized Born end state Systems.

The end state (complex, receptor, ligand) Systems are created by taking a
subset of the explicit solvent System that was simulated, ensuring that the
molecular mechanics parameters exactly match those used for sampling, and
adding generalized Born forces to them.
"""

import copy
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import openmm
from openff.units import Quantity, unit
from openmm import app
from openmm.app.internal.customgbforces import (
    GBSAGBn2Force,
    GBSAGBnForce,
    GBSAHCTForce,
    GBSAOBC1Force,
    GBSAOBC2Force,
)

GB_MODELS = {
    "HCT": GBSAHCTForce,
    "OBC1": GBSAOBC1Force,
    "OBC2": GBSAOBC2Force,
    "GBn": GBSAGBnForce,
    "GBn2": GBSAGBn2Force,
}

# Force groups used to decompose the end state energies
FORCE_GROUPS = {
    "bond": 0,
    "angle": 1,
    "torsion": 2,
    "elec": 3,
    "vdw": 4,
    "gb_polar": 5,
    "gb_nonpolar": 6,
}

# Forces which can be safely ignored when computing potential energies
_IGNORED_FORCES = (
    openmm.CMMotionRemover,
    openmm.MonteCarloBarostat,
    openmm.MonteCarloAnisotropicBarostat,
    openmm.MonteCarloMembraneBarostat,
    openmm.MonteCarloFlexibleBarostat,
)


def subset_topology(topology: app.Topology, atom_indices: Sequence[int]) -> app.Topology:
    """
    Create a new Topology containing only the selected atoms, in the
    order given by ``atom_indices``.

    Chains & residues are recreated as necessary, a new residue is started
    whenever consecutive selected atoms come from different residues.
    Bonds between selected atoms are retained.

    Parameters
    ----------
    topology : openmm.app.Topology
      The Topology to subset.
    atom_indices : Sequence[int]
      The indices of the atoms to retain.

    Returns
    -------
    openmm.app.Topology
      The subset Topology, with no periodic box vectors.
    """
    atoms = list(topology.atoms())
    new_top = app.Topology()
    old_to_new = {}

    current_chain = None
    current_residue = None
    old_chain = None
    old_residue = None

    for idx in atom_indices:
        atom = atoms[idx]
        if atom.residue.chain is not old_chain:
            old_chain = atom.residue.chain
            current_chain = new_top.addChain(id=old_chain.id)
            old_residue = None
        if atom.residue is not old_residue:
            old_residue = atom.residue
            current_residue = new_top.addResidue(
                old_residue.name,
                current_chain,
                id=old_residue.id,
                insertionCode=old_residue.insertionCode,
            )
        old_to_new[idx] = new_top.addAtom(atom.name, atom.element, current_residue, id=atom.id)

    for bond in topology.bonds():
        a1, a2 = bond[0].index, bond[1].index
        if a1 in old_to_new and a2 in old_to_new:
            new_top.addBond(old_to_new[a1], old_to_new[a2], type=bond.type, order=bond.order)

    return new_top


def subset_system(
    system: openmm.System,
    atom_indices: Sequence[int],
) -> openmm.System:
    """
    Create a non-periodic System containing only the selected atoms of an
    input System.

    Bonded terms are retained if all their atoms are in the subset and the
    NonbondedForce is recreated with no cutoff. Constraints, barostats and
    center of mass motion removers are dropped since they do not
    contribute to the potential energy.

    Parameters
    ----------
    system : openmm.System
      The System to subset.
    atom_indices : Sequence[int]
      The indices of the atoms to retain, the new System will have its
      atoms in this order.
    Returns
    -------
    openmm.System
      The new non-periodic System.

    Raises
    ------
    ValueError
      If the System contains virtual sites or forces which are not supported.
    """
    atom_indices = [int(i) for i in atom_indices]
    if len(set(atom_indices)) != len(atom_indices):
        raise ValueError("Duplicate atom indices passed to subset_system")

    # map of the input System atom index to new System index
    old_to_new = {old: new for new, old in enumerate(atom_indices)}

    new_system = openmm.System()
    for old in atom_indices:
        if system.isVirtualSite(old):
            raise ValueError(f"Virtual sites are not supported, atom {old} is a virtual site")
        new_system.addParticle(system.getParticleMass(old))

    def _remap(indices):
        try:
            return [old_to_new[i] for i in indices]
        except KeyError:
            return None

    for force in system.getForces():
        if isinstance(force, _IGNORED_FORCES):
            continue
        elif isinstance(force, openmm.HarmonicBondForce):
            new_force = openmm.HarmonicBondForce()
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                if (new := _remap([p1, p2])) is not None:
                    new_force.addBond(*new, length, k)
        elif isinstance(force, openmm.HarmonicAngleForce):
            new_force = openmm.HarmonicAngleForce()
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                if (new := _remap([p1, p2, p3])) is not None:
                    new_force.addAngle(*new, angle, k)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            new_force = openmm.PeriodicTorsionForce()
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = force.getTorsionParameters(i)
                if (new := _remap([p1, p2, p3, p4])) is not None:
                    new_force.addTorsion(*new, per, phase, k)
        elif isinstance(force, openmm.RBTorsionForce):
            new_force = openmm.RBTorsionForce()
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, *cs = force.getTorsionParameters(i)
                if (new := _remap([p1, p2, p3, p4])) is not None:
                    new_force.addTorsion(*new, *cs)
        elif isinstance(force, openmm.CMAPTorsionForce):
            new_force = openmm.CMAPTorsionForce()
            for i in range(force.getNumMaps()):
                new_force.addMap(*force.getMapParameters(i))
            for i in range(force.getNumTorsions()):
                cmap, *atoms = force.getTorsionParameters(i)
                if (new := _remap(atoms)) is not None:
                    new_force.addTorsion(cmap, *new)
        elif isinstance(force, openmm.NonbondedForce):
            new_force = _subset_nonbonded(force, atom_indices, old_to_new)
        else:
            errmsg = f"Unsupported force for MM/GBSA: {force.__class__.__name__}"
            raise ValueError(errmsg)

        new_force.setForceGroup(force.getForceGroup())
        new_system.addForce(new_force)

    return new_system


def _subset_nonbonded(
    force: openmm.NonbondedForce,
    source_indices: list[int],
    old_to_new: dict[int, int],
) -> openmm.NonbondedForce:
    """
    Recreate a NonbondedForce for a subset of atoms with no cutoff.
    """
    if force.getNumParticleParameterOffsets() > 0 or force.getNumExceptionParameterOffsets() > 0:
        raise ValueError("NonbondedForce parameter offsets are not supported")

    new_force = openmm.NonbondedForce()
    new_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    new_force.setUseDispersionCorrection(False)

    for old in source_indices:
        new_force.addParticle(*force.getParticleParameters(old))

    for i in range(force.getNumExceptions()):
        p1, p2, qq, sigma, eps = force.getExceptionParameters(i)
        if p1 in old_to_new and p2 in old_to_new:
            new_force.addException(old_to_new[p1], old_to_new[p2], qq, sigma, eps)

    return new_force


def get_debye_kappa(
    salt_concentration: Quantity,
    solvent_dielectric: float,
    temperature: Quantity,
) -> float:
    """
    Get the Debye-Huckel screening parameter in 1/nm.

    This follows the same conventions as Amber & OpenMM, including the 0.73
    factor accounting for ion exclusion.
    """
    conc = salt_concentration.to(unit.molar).m
    temp = temperature.to(unit.kelvin).m
    if conc == 0:
        return 0.0
    return 50.33355 * math.sqrt(conc / solvent_dielectric / temp) * 7.3


def _get_ace_expression(gamma: Quantity, probe_radius: Quantity, offset: float) -> str:
    """
    Get the ACE non-polar solvation energy expression for a CustomGBForce.

    gamma is the surface tension, the energy is
    4 pi gamma (r + probe)^2 (r / B)^6.
    """
    gamma_kj_nm2 = gamma.to(unit.kilojoule_per_mole / unit.nanometer**2).m
    probe = probe_radius.to(unit.nanometer).m
    return (
        f"{4 * math.pi * gamma_kj_nm2:.16g}*(radius+{probe:.16g})^2*(radius/B)^6;"
        f"radius=or+{offset:.16g}"
    )


def get_gb_forces(
    topology: app.Topology,
    charges: npt.NDArray,
    gb_model: str,
    solute_dielectric: float,
    solvent_dielectric: float,
    kappa: float,
    surface_tension: Quantity,
    probe_radius: Quantity,
) -> tuple[openmm.CustomGBForce, openmm.CustomGBForce]:
    """
    Create the polar and non-polar (ACE) generalized Born forces.

    Two separate CustomGBForces are created so that the polar and non-polar
    contributions can be evaluated independently. Both share the same Born
    radii definition.

    Parameters
    ----------
    topology : openmm.app.Topology
      The Topology of the System, must contain elements and bonds,
      these are used to assign the GB radii and screening parameters.
    charges : npt.NDArray
      The partial charges for each atom (elementary charge units).
    gb_model : str
      The name of the GB model, one of ``GB_MODELS``.
    solute_dielectric : float
      The solute dielectric.
    solvent_dielectric : float
      The solvent dielectric.
    kappa : float
      The Debye-Huckel screening parameter in 1/nm.
    surface_tension : openff.units.Quantity
      The surface tension used for the ACE term.
    probe_radius : openff.units.Quantity
      The solvent probe radius used for the ACE term.

    Returns
    -------
    polar_force : openmm.CustomGBForce
      The polar GB force.
    nonpolar_force : openmm.CustomGBForce
      The non-polar ACE force.
    """
    if topology.getNumAtoms() != len(charges):
        raise ValueError("Number of charges does not match number of atoms in topology")

    for atom in topology.atoms():
        if atom.element is None:
            raise ValueError(f"Atom {atom} has no element, cannot assign GB radii")

    force_cls = GB_MODELS[gb_model]
    polar = force_cls(
        solventDielectric=solvent_dielectric,
        soluteDielectric=solute_dielectric,
        SA=None,
        cutoff=None,
        kappa=kappa,
    )

    params = np.asarray(force_cls.getStandardParameters(topology), dtype=float)
    polar.addParticles(np.column_stack([np.asarray(charges, dtype=float), params]))
    polar.finalize()
    polar.setNonbondedMethod(openmm.CustomGBForce.NoCutoff)

    # The non-polar force reuses the computed values (i.e. Born radii) of
    # the polar force but replaces the energy terms with just the ACE term
    nonpolar = copy.deepcopy(polar)
    for i in range(nonpolar.getNumEnergyTerms()):
        nonpolar.setEnergyTermParameters(i, "0", openmm.CustomGBForce.SingleParticle)
    nonpolar.setEnergyTermParameters(
        0,
        _get_ace_expression(surface_tension, probe_radius, force_cls.OFFSET),
        openmm.CustomGBForce.SingleParticle,
    )

    return polar, nonpolar


def _assign_decomposition_groups(system: openmm.System) -> None:
    """
    Assign the force groups and split the NonbondedForce into separate
    electrostatic and van der Waals NonbondedForces.
    """
    nonbonded_indices = [
        i for i, f in enumerate(system.getForces()) if isinstance(f, openmm.NonbondedForce)
    ]
    if len(nonbonded_indices) != 1:
        raise ValueError("Expected exactly one NonbondedForce")
    nonbonded = system.getForce(nonbonded_indices[0])
    elec = copy.deepcopy(nonbonded)
    vdw = copy.deepcopy(nonbonded)
    system.removeForce(nonbonded_indices[0])

    for i in range(elec.getNumParticles()):
        q, sigma, eps = elec.getParticleParameters(i)
        elec.setParticleParameters(i, q, sigma, 0.0)
        vdw.setParticleParameters(i, 0.0, sigma, eps)
    for i in range(elec.getNumExceptions()):
        p1, p2, qq, sigma, eps = elec.getExceptionParameters(i)
        elec.setExceptionParameters(i, p1, p2, qq, sigma, 0.0)
        vdw.setExceptionParameters(i, p1, p2, 0.0, sigma, eps)

    elec.setForceGroup(FORCE_GROUPS["elec"])
    vdw.setForceGroup(FORCE_GROUPS["vdw"])
    system.addForce(elec)
    system.addForce(vdw)

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            force.setForceGroup(FORCE_GROUPS["bond"])
        elif isinstance(force, openmm.HarmonicAngleForce):
            force.setForceGroup(FORCE_GROUPS["angle"])
        elif isinstance(
            force, (openmm.PeriodicTorsionForce, openmm.RBTorsionForce, openmm.CMAPTorsionForce)
        ):
            force.setForceGroup(FORCE_GROUPS["torsion"])


def get_charges(system: openmm.System) -> npt.NDArray:
    """
    Get the partial charges, in units of elementary charge, from the
    single NonbondedForce of a System.
    """
    nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if len(nonbonded) != 1:
        raise ValueError("Expected exactly one NonbondedForce")
    return np.array(
        [
            nonbonded[0].getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge)
            for i in range(nonbonded[0].getNumParticles())
        ]
    )


def create_gb_system(
    system: openmm.System,
    topology: app.Topology,
    atom_indices: Sequence[int],
    gb_model: str,
    solute_dielectric: float,
    solvent_dielectric: float,
    kappa: float,
    surface_tension: Quantity,
    probe_radius: Quantity,
) -> tuple[openmm.System, app.Topology]:
    """
    Create a decomposed generalized Born System for a subset of atoms
    of an explicit solvent System.

    Each energy component is placed in its own force group, as defined
    by ``FORCE_GROUPS``.

    Parameters
    ----------
    system : openmm.System
      The explicit solvent System.
    topology : openmm.app.Topology
      The Topology for the explicit solvent System.
    atom_indices : Sequence[int]
      The atoms of ``system`` to include, in the order they should appear
      in the new System.
    gb_model : str
      The name of the GB model, one of ``GB_MODELS``.
    solute_dielectric : float
      The solute dielectric.
    solvent_dielectric : float
      The solvent dielectric.
    kappa : float
      The Debye-Huckel screening parameter in 1/nm.
    surface_tension : openff.units.Quantity
      The surface tension used for the ACE term.
    probe_radius : openff.units.Quantity
      The solvent probe radius used for the ACE term.

    Returns
    -------
    gb_system : openmm.System
      The generalized Born System.
    gb_topology : openmm.app.Topology
      The Topology for the generalized Born System.
    """
    gb_system = subset_system(system, atom_indices)
    gb_topology = subset_topology(topology, atom_indices)
    charges = get_charges(gb_system)

    _assign_decomposition_groups(gb_system)

    polar, nonpolar = get_gb_forces(
        topology=gb_topology,
        charges=charges,
        gb_model=gb_model,
        solute_dielectric=solute_dielectric,
        solvent_dielectric=solvent_dielectric,
        kappa=kappa,
        surface_tension=surface_tension,
        probe_radius=probe_radius,
    )
    polar.setForceGroup(FORCE_GROUPS["gb_polar"])
    nonpolar.setForceGroup(FORCE_GROUPS["gb_nonpolar"])
    gb_system.addForce(polar)
    gb_system.addForce(nonpolar)

    return gb_system, gb_topology

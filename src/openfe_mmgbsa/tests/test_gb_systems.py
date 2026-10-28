import copy

import numpy as np
import openmm
import pytest
from numpy.testing import assert_allclose
from openff.units import unit
from openmm import app
from openmm import unit as omm_unit

from openfe_mmgbsa.gb_systems import (
    FORCE_GROUPS,
    GB_MODELS,
    create_gb_system,
    get_debye_kappa,
    subset_system,
    subset_topology,
)

IMPLICIT_XML = {
    "HCT": "implicit/hct.xml",
    "OBC1": "implicit/obc1.xml",
    "OBC2": "implicit/obc2.xml",
    "GBn": "implicit/gbn.xml",
    "GBn2": "implicit/gbn2.xml",
}

DEFAULT_GAMMA = 0.0054 * unit.kilocalorie_per_mole / unit.angstrom**2
DEFAULT_PROBE = 0.14 * unit.nanometer


def _energy(system, positions, platform, groups=-1):
    integrator = openmm.VerletIntegrator(1.0)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    e = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
    del context, integrator
    return e.value_in_unit(omm_unit.kilojoule_per_mole)


@pytest.fixture(scope="module")
def T4_vacuum(T4_pdb):
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(T4_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    return system


@pytest.fixture(scope="module")
def T4_solvated():
    """A small solvated T4 system."""
    from importlib import resources

    with resources.as_file(resources.files("openfe.tests.data")) as d:
        pdb = app.PDBFile(str(d / "181l_only.pdb"))
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(ff, padding=0.8 * omm_unit.nanometer, ionicStrength=0.1 * omm_unit.molar)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * omm_unit.nanometer,
        constraints=app.HBonds,
    )
    system.addForce(openmm.MonteCarloBarostat(1 * omm_unit.bar, 298.15 * omm_unit.kelvin))
    return system, modeller.topology, modeller.positions, pdb.topology.getNumAtoms()


def test_debye_kappa():
    # Matches the OpenMM AmberPrmtopFile conversion
    kappa = get_debye_kappa(0.15 * unit.molar, 78.5, 298.15 * unit.kelvin)
    expected = 50.33355 * np.sqrt(0.15 / 78.5 / 298.15) * 7.3
    assert kappa == pytest.approx(expected)
    assert get_debye_kappa(0 * unit.molar, 78.5, 298.15 * unit.kelvin) == 0.0


def test_subset_topology(T4_pdb):
    # ACE cap atoms (C, O, CH3, 1HH3, 2HH3) in a scrambled order
    indices = [4, 2, 3, 0, 1]
    top = subset_topology(T4_pdb.topology, indices)
    atoms = list(T4_pdb.topology.atoms())
    new_atoms = list(top.atoms())
    assert [a.name for a in new_atoms] == [atoms[i].name for i in indices]
    # bonds within the subset are retained
    expected = {
        frozenset((a.index, b.index))
        for a, b in T4_pdb.topology.bonds()
        if a.index in indices and b.index in indices
    }
    found = {frozenset((indices[a.index], indices[b.index])) for a, b in top.bonds()}
    assert found == expected
    assert len(found) > 0


def test_subset_matches_direct_vacuum(T4_solvated, T4_vacuum, T4_pdb, reference_platform):
    """
    Subsetting the protein out of a solvated PME System gives the
    same energies as directly creating a vacuum System (with the same
    constraints, since constrained bonds have no HarmonicBond term).
    """
    system, topology, positions, n_protein = T4_solvated
    sub = subset_system(system, list(range(n_protein)))
    assert sub.getNumParticles() == n_protein
    assert not sub.usesPeriodicBoundaryConditions()

    ff = app.ForceField("amber14-all.xml")
    direct = ff.createSystem(
        T4_pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
    )
    pos = positions[:n_protein]
    assert_allclose(
        _energy(sub, pos, reference_platform), _energy(direct, pos, reference_platform), rtol=1e-6
    )


def test_subset_rejects_unsupported_force(T4_vacuum):
    system = copy.deepcopy(T4_vacuum)
    system.addForce(openmm.CustomExternalForce("x"))
    with pytest.raises(ValueError, match="Unsupported force"):
        subset_system(system, [0, 1, 2])


def test_subset_rejects_duplicates(T4_vacuum):
    with pytest.raises(ValueError, match="Duplicate"):
        subset_system(T4_vacuum, [0, 1, 1])


@pytest.mark.parametrize("gb_model", list(GB_MODELS))
@pytest.mark.parametrize("salt", [0.0, 0.15])
def test_gb_matches_openmm_implicit(gb_model, salt, T4_pdb, T4_vacuum, reference_platform):
    """
    Our decomposed GB System gives the same total energy as OpenMM's own
    implicit solvent ForceField files (which use the ACE term with the
    default surface tension).
    """
    n_atoms = T4_pdb.topology.getNumAtoms()
    kappa = get_debye_kappa(salt * unit.molar, 78.5, 298.15 * unit.kelvin)

    gb_system, gb_top = create_gb_system(
        system=T4_vacuum,
        topology=T4_pdb.topology,
        atom_indices=list(range(n_atoms)),
        gb_model=gb_model,
        solute_dielectric=1.0,
        solvent_dielectric=78.5,
        kappa=kappa,
        surface_tension=DEFAULT_GAMMA,
        probe_radius=DEFAULT_PROBE,
    )

    ff = app.ForceField("amber14-all.xml", IMPLICIT_XML[gb_model])
    reference = ff.createSystem(
        T4_pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        implicitSolventKappa=kappa / omm_unit.nanometer,
    )

    ours = _energy(gb_system, T4_pdb.positions, reference_platform)
    ref = _energy(reference, T4_pdb.positions, reference_platform)
    assert_allclose(ours, ref, rtol=1e-6)


@pytest.mark.parametrize("gb_model", list(GB_MODELS))
def test_decomposition_sums_to_total(gb_model, T4_pdb, T4_vacuum, reference_platform):
    n_atoms = T4_pdb.topology.getNumAtoms()
    gb_system, _ = create_gb_system(
        system=T4_vacuum,
        topology=T4_pdb.topology,
        atom_indices=list(range(n_atoms)),
        gb_model=gb_model,
        solute_dielectric=1.0,
        solvent_dielectric=78.5,
        kappa=0.0,
        surface_tension=DEFAULT_GAMMA,
        probe_radius=DEFAULT_PROBE,
    )
    # every force is in one of the decomposition groups
    assert {f.getForceGroup() for f in gb_system.getForces()} == set(FORCE_GROUPS.values())

    total = _energy(gb_system, T4_pdb.positions, reference_platform)
    components = [
        _energy(gb_system, T4_pdb.positions, reference_platform, groups={g})
        for g in FORCE_GROUPS.values()
    ]
    assert_allclose(sum(components), total, rtol=1e-6)

    # The vacuum MM energy is unchanged by the decomposition
    mm = sum(
        _energy(gb_system, T4_pdb.positions, reference_platform, groups={FORCE_GROUPS[c]})
        for c in ["bond", "angle", "torsion", "elec", "vdw"]
    )
    assert_allclose(mm, _energy(T4_vacuum, T4_pdb.positions, reference_platform), rtol=1e-6)


def test_surface_tension_scales_nonpolar(T4_pdb, T4_vacuum, reference_platform):
    n_atoms = T4_pdb.topology.getNumAtoms()

    def _nonpolar(gamma):
        system, _ = create_gb_system(
            system=T4_vacuum,
            topology=T4_pdb.topology,
            atom_indices=list(range(n_atoms)),
            gb_model="OBC2",
            solute_dielectric=1.0,
            solvent_dielectric=78.5,
            kappa=0.0,
            surface_tension=gamma,
            probe_radius=DEFAULT_PROBE,
        )
        return _energy(system, T4_pdb.positions, reference_platform, groups={FORCE_GROUPS["gb_nonpolar"]})

    e1 = _nonpolar(DEFAULT_GAMMA)
    e2 = _nonpolar(2 * DEFAULT_GAMMA)
    assert e1 > 0
    assert_allclose(e2, 2 * e1, rtol=1e-6)
    assert _nonpolar(0 * DEFAULT_GAMMA) == pytest.approx(0.0)

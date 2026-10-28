import copy
import dataclasses

import gufe
import numpy as np
import openfe
import openmm
import pytest
from gufe.protocols import execute_DAG
from numpy.testing import assert_allclose
from openff.units import unit
from openmm import unit as omm_unit
from pydantic import ValidationError

from openfe_mmgbsa import (
    MMGBSAAnalysisUnit,
    MMGBSAProtocol,
    MMGBSAProtocolResult,
    MMGBSAProtocolSettings,
    MMGBSASettings,
    MMGBSASetupUnit,
    MMGBSASimulationUnit,
)
from openfe_mmgbsa.atom_indices import (
    check_identical_water_parameters,
    check_waters_in_trajectory,
    get_mmgbsa_atom_indices,
)
from openfe_mmgbsa.gb_systems import FORCE_GROUPS
from openfe_mmgbsa.mmgbsa_units import get_end_state_atoms


@pytest.fixture
def settings():
    s = MMGBSAProtocol.default_settings()
    s.engine_settings.compute_platform = "CPU"
    s.protocol_repeats = 2
    return s


def test_default_settings():
    s = MMGBSAProtocol.default_settings()
    assert isinstance(s, MMGBSAProtocolSettings)
    assert s.mmgbsa_settings == MMGBSASettings()
    assert s.output_settings.output_indices == "not water"


def test_protocol_serialization_roundtrip(settings):
    protocol = MMGBSAProtocol(settings=settings)
    roundtrip = gufe.tokenization.GufeTokenizable.from_dict(protocol.to_dict())
    assert roundtrip == protocol
    assert roundtrip.settings.mmgbsa_settings.gb_model == "OBC2"


@pytest.mark.parametrize(
    "field, value",
    [
        ("solute_dielectric", 0),
        ("solvent_dielectric", -1),
        ("n_waters", -1),
        ("frame_stride", 0),
        ("salt_concentration", -0.1 * unit.molar),
        ("gb_model", "GBfoo"),
        ("analysis_start_time", -1 * unit.nanosecond),
    ],
)
def test_bad_mmgbsa_settings(field, value):
    with pytest.raises(ValidationError):
        MMGBSASettings(**{field: value})


def test_surface_tension_units():
    s = MMGBSASettings(surface_tension=0.02259 * unit.kilojoule_per_mole / unit.angstrom**2)
    assert s.surface_tension.to("kilocalorie_per_mole / angstrom**2").m == pytest.approx(
        0.0054, rel=1e-3
    )
    with pytest.raises(ValidationError):
        MMGBSASettings(surface_tension=1 * unit.nanometer)


def test_validate_no_protein(toluene, settings):
    system = gufe.ChemicalSystem({"ligand": toluene, "solvent": openfe.SolventComponent()})
    with pytest.raises(ValueError, match="ProteinComponent"):
        MMGBSAProtocol(settings).validate(stateA=system, stateB=system, mapping=None)


def test_validate_two_ligands(benzene_modifications, T4_protein_component, settings):
    system = gufe.ChemicalSystem(
        {
            "ligand": benzene_modifications["toluene"],
            "ligand2": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": openfe.SolventComponent(),
        }
    )
    with pytest.raises(ValueError, match="exactly one SmallMoleculeComponent"):
        MMGBSAProtocol(settings).validate(stateA=system, stateB=system, mapping=None)


def test_validate_no_solvent(toluene, T4_protein_component, settings):
    system = gufe.ChemicalSystem({"ligand": toluene, "protein": T4_protein_component})
    with pytest.raises(ValueError, match="explicitly solvated"):
        MMGBSAProtocol(settings).validate(stateA=system, stateB=system, mapping=None)


def test_validate_nwat_requires_water_output(toluene_complex_system, settings):
    settings.mmgbsa_settings.n_waters = 5
    with pytest.raises(ValueError, match="does not select water"):
        MMGBSAProtocol(settings).validate(
            stateA=toluene_complex_system, stateB=toluene_complex_system, mapping=None
        )

    settings.output_settings.output_indices = "all"
    MMGBSAProtocol(settings).validate(
        stateA=toluene_complex_system, stateB=toluene_complex_system, mapping=None
    )


def test_validate_different_states(toluene_complex_system, benzene_modifications, T4_protein_component, settings):
    other = gufe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": openfe.SolventComponent(),
        }
    )
    with pytest.raises(ValueError, match="do not match"):
        MMGBSAProtocol(settings).validate(stateA=toluene_complex_system, stateB=other, mapping=None)


def test_create(toluene_complex_system, settings):
    protocol = MMGBSAProtocol(settings=settings)
    dag = protocol.create(stateA=toluene_complex_system, stateB=toluene_complex_system, mapping=None)
    units = list(dag.protocol_units)
    assert len(units) == 6
    assert sum(isinstance(u, MMGBSASetupUnit) for u in units) == 2
    assert sum(isinstance(u, MMGBSASimulationUnit) for u in units) == 2
    analysis = [u for u in units if isinstance(u, MMGBSAAnalysisUnit)]
    assert len(analysis) == 2
    for u in analysis:
        assert isinstance(u.inputs["setup_results"], MMGBSASetupUnit)
        assert isinstance(u.inputs["simulation_results"], MMGBSASimulationUnit)
        assert u.inputs["setup_results"].inputs["repeat_id"] == u.inputs["repeat_id"]
        assert "MMGBSA Analysis: T4_protein:toluene" in u.name


@pytest.fixture(scope="module")
def dry_setup(tmp_path_factory, toluene, T4_protein_component):
    """Dry run of the setup unit for T4 lysozyme + toluene with NWAT."""
    settings = MMGBSAProtocol.default_settings()
    settings.engine_settings.compute_platform = "CPU"
    settings.protocol_repeats = 1
    settings.mmgbsa_settings.n_waters = 4
    settings.output_settings.output_indices = "all"
    system = gufe.ChemicalSystem(
        {
            "ligand": toluene,
            "protein": T4_protein_component,
            "solvent": openfe.SolventComponent(),
        }
    )
    protocol = MMGBSAProtocol(settings=settings)
    dag = protocol.create(stateA=system, stateB=system, mapping=None)
    setup = [u for u in dag.protocol_units if isinstance(u, MMGBSASetupUnit)][0]
    shared = tmp_path_factory.mktemp("shared")
    outputs = setup.run(dry=True, verbose=False, shared_basepath=shared, scratch_basepath=shared)
    return outputs, toluene, T4_protein_component


def test_dry_setup_indices(dry_setup):
    outputs, toluene, protein = dry_setup
    debug = outputs["debug"]
    indices = debug["mmgbsa_atom_indices"]
    topology = debug["topology_from_pdb"]
    n_protein = protein.to_openmm_topology().getNumAtoms()

    assert indices.receptor == list(range(n_protein))
    assert indices.ligand == list(range(n_protein, n_protein + toluene.to_openff().n_atoms))
    assert len(indices.waters) > 1000
    # all atoms are written with output_indices="all"
    assert indices.trajectory == list(range(topology.getNumAtoms()))
    atoms = list(topology.atoms())
    for water in indices.waters:
        assert [atoms[i].element.symbol for i in water] == ["O", "H", "H"]
    assert outputs["mmgbsa_atom_indices"].is_file()


def test_dry_setup_gb_systems(dry_setup):
    outputs, toluene, protein = dry_setup
    debug = outputs["debug"]
    indices = debug["mmgbsa_atom_indices"]
    gb_systems = debug["gb_systems"]
    n_rec = len(indices.receptor) + 4 * 3
    n_lig = len(indices.ligand)

    assert gb_systems["receptor"].getNumParticles() == n_rec
    assert gb_systems["ligand"].getNumParticles() == n_lig
    assert gb_systems["complex"].getNumParticles() == n_rec + n_lig
    for system in gb_systems.values():
        assert not system.usesPeriodicBoundaryConditions()
        assert {f.getForceGroup() for f in system.getForces()} == set(FORCE_GROUPS.values())


def test_dry_setup_single_trajectory_cancellation(dry_setup, reference_platform):
    """
    Bonded terms exactly cancel, and the complex vacuum interaction energy is
    the sum of the receptor/ligand energies when they are far apart.
    """
    outputs, _, _ = dry_setup
    debug = outputs["debug"]
    indices = debug["mmgbsa_atom_indices"]
    gb_systems = debug["gb_systems"]
    positions = np.load(outputs["positions"])
    atoms = get_end_state_atoms(indices, 4)

    def _energies(state, pos):
        integrator = openmm.VerletIntegrator(1.0)
        context = openmm.Context(gb_systems[state], integrator, reference_platform)
        context.setPositions(pos * omm_unit.nanometer)
        e = {
            name: context.getState(getEnergy=True, groups={g})
            .getPotentialEnergy()
            .value_in_unit(omm_unit.kilojoule_per_mole)
            for name, g in FORCE_GROUPS.items()
        }
        del context, integrator
        return e

    e = {state: _energies(state, positions[atoms[state]]) for state in atoms}
    for comp in ["bond", "angle", "torsion"]:
        assert_allclose(e["complex"][comp], e["receptor"][comp] + e["ligand"][comp], atol=1e-6)

    # move the ligand far away, all interactions vanish
    far = positions[atoms["complex"]].copy()
    far[-len(indices.ligand):] += 1000.0
    e_far = _energies("complex", far)
    for comp in FORCE_GROUPS:
        assert_allclose(
            e_far[comp], e["receptor"][comp] + e["ligand"][comp], atol=1e-3
        )


def test_end_state_atoms_too_many_waters(dry_setup):
    outputs, _, _ = dry_setup
    indices = outputs["debug"]["mmgbsa_atom_indices"]
    with pytest.raises(ValueError, match="only"):
        get_end_state_atoms(indices, len(indices.waters) + 1)


@pytest.mark.parametrize(
    "start_time, expected",
    [(None, 0), (0, 0), (20, 0), (21, 1), (40, 1), (100, 4)],
)
def test_analysis_start_frame(settings, start_time, expected):
    # frame i is written at production time (i + 1) * 20 ps
    settings.output_settings.trajectory_write_interval = 20 * unit.picosecond
    if start_time is not None:
        start_time = start_time * unit.picosecond
    settings.mmgbsa_settings.analysis_start_time = start_time
    assert MMGBSAAnalysisUnit._get_start_frame(settings) == expected


def test_atom_indices_selection_missing_solute(dry_setup):
    outputs, toluene, protein = dry_setup
    topology = outputs["debug"]["topology_from_pdb"]
    with pytest.raises(ValueError, match="does not include"):
        get_mmgbsa_atom_indices(topology, protein, toluene, "not protein")


def test_atom_indices_nwat_checks(dry_setup):
    outputs, _, _ = dry_setup
    indices = outputs["debug"]["mmgbsa_atom_indices"]
    system = outputs["debug"]["system"]

    check_identical_water_parameters(system, indices.waters)
    no_waters = dataclasses.replace(
        indices, trajectory=indices.receptor + indices.ligand
    )
    with pytest.raises(ValueError, match="water molecules"):
        check_waters_in_trajectory(no_waters, "not water")

    # a water with modified charges
    system = copy.deepcopy(system)
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    idx = indices.waters[-1][0]
    q, sig, eps = nb.getParticleParameters(idx)
    nb.setParticleParameters(idx, q * 1.1, sig, eps)
    with pytest.raises(ValueError, match="identical parameters"):
        check_identical_water_parameters(system, indices.waters)


def _fake_result(summaries):
    return MMGBSAProtocolResult(
        **{
            str(i): [
                gufe.ProtocolUnitResult(
                    source_key=f"unit{i}",
                    inputs={},
                    outputs={"summary": s, "per_frame_energies": f"{i}.csv"},
                )
            ]
            for i, s in enumerate(summaries)
        }
    )


def _summary(total, sem):
    return {
        "total": {"mean": total, "sem": sem},
        "vdw": {"mean": 2 * total, "sem": 2 * sem},
    }


def test_result_single_repeat():
    result = _fake_result([_summary(-10.0, 0.5)])
    assert result.get_estimate().m_as("kcal/mol") == pytest.approx(-10.0)
    assert result.get_uncertainty().m_as("kcal/mol") == pytest.approx(0.5)
    assert result.get_individual_estimates()[0][1].m_as("kcal/mol") == pytest.approx(0.5)
    vdw = result.get_components()["vdw"]
    assert vdw[0].m_as("kcal/mol") == pytest.approx(-20.0)
    assert vdw[1].m_as("kcal/mol") == pytest.approx(1.0)
    assert result.get_per_frame_energies_filenames() == ["0.csv"]


def test_result_multiple_repeats():
    result = _fake_result([_summary(-10.0, 0.5), _summary(-12.0, 0.5)])
    assert result.get_estimate().m_as("kcal/mol") == pytest.approx(-11.0)
    assert result.get_uncertainty().m_as("kcal/mol") == pytest.approx(np.std([-10, -12], ddof=1))
    assert len(result.get_individual_estimates()) == 2
    vdw = result.get_components()["vdw"]
    assert vdw[0].m_as("kcal/mol") == pytest.approx(-22.0)
    assert vdw[1].m_as("kcal/mol") == pytest.approx(np.std([-20, -24], ddof=1))


@pytest.mark.slow
def test_protocol_execution_nwat(tmp_path, toluene, T4_protein_component):
    """A very short end-to-end run of MM/GBSA-NWAT."""
    s = MMGBSAProtocol.default_settings()
    s.protocol_repeats = 1
    s.engine_settings.compute_platform = "CPU"
    s.solvation_settings.solvent_padding = 1.0 * unit.nanometer
    s.simulation_settings.minimization_steps = 100
    s.simulation_settings.equilibration_length_nvt = 8 * unit.picosecond
    s.simulation_settings.equilibration_length = 8 * unit.picosecond
    s.simulation_settings.production_length = 20 * unit.picosecond
    s.output_settings.checkpoint_interval = 4 * unit.picosecond
    s.output_settings.trajectory_write_interval = 4 * unit.picosecond
    s.output_settings.output_indices = "all"
    s.mmgbsa_settings.n_waters = 4
    s.mmgbsa_settings.analysis_start_time = 8 * unit.picosecond

    system = gufe.ChemicalSystem(
        {"protein": T4_protein_component, "ligand": toluene, "solvent": openfe.SolventComponent()}
    )
    protocol = MMGBSAProtocol(settings=s)
    dag = protocol.create(stateA=system, stateB=system, mapping=None)
    shared, scratch = tmp_path / "shared", tmp_path / "scratch"
    shared.mkdir()
    scratch.mkdir()
    dag_result = execute_DAG(
        dag, shared_basedir=shared, scratch_basedir=scratch, keep_shared=True, raise_error=True
    )
    assert dag_result.ok()
    result = protocol.gather([dag_result])

    # 5 frames at 4, 8, ... 20 ps, the first is skipped
    rows = result.get_per_frame_energies()[0]
    assert [r["frame"] for r in rows] == [1, 2, 3, 4]
    assert all(len(r["water_ids"]) == 4 for r in rows)

    components = result.get_components()
    for comp in ["bond", "angle", "torsion", "bonded"]:
        assert components[comp][0].m_as("kcal/mol") == pytest.approx(0.0, abs=1e-6)
    # toluene binds T4 L99A through vdW interactions
    assert components["vdw"][0].m_as("kcal/mol") < 0
    assert components["total"][0] == result.get_estimate()
    assert result.get_uncertainty().m_as("kcal/mol") > 0

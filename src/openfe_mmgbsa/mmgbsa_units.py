"""ProtocolUnits for the OpenMM MM/GBSA Protocol.

The setup and simulation units extend the OpenFE PlainMD units, the
analysis unit re-scores the production trajectory with MM/GBSA.
"""

import logging
import math
import pathlib
from typing import Any

import gufe
import numpy as np
import openmm
from gufe import ChemicalSystem
from openfe.protocols.openmm_md.plain_md_methods import (
    PlainMDSetupUnit,
    PlainMDSimulationUnit,
    PlainMDUnitMixin,
)
from openfe.protocols.openmm_utils import omm_compute, serialization, system_validation
from openfe.utils import log_system_probe
from openmm import app

from .atom_indices import (
    MMGBSAAtomIndices,
    check_identical_water_parameters,
    check_waters_in_trajectory,
    get_mmgbsa_atom_indices,
)
from .frames import get_universe, iterate_frames
from .gb_systems import create_gb_system, get_debye_kappa
from .mmgbsa_settings import MMGBSAProtocolSettings
from .scoring import STATES, EndStateEvaluator, score_frames, summarize, write_rows

logger = logging.getLogger(__name__)


def get_end_state_atoms(indices: MMGBSAAtomIndices, n_waters: int) -> dict[str, list[int]]:
    """
    Get the atoms, in the full System, of each MM/GBSA end state.

    The first ``n_waters`` waters are used as "slots" which are filled with
    the closest waters to the ligand in each frame.

    Returns
    -------
    dict[str, list[int]]
      The atom indices of the complex (receptor, waters, ligand),
      receptor (receptor, waters) and ligand states.
    """
    if n_waters > len(indices.waters):
        errmsg = f"Requested {n_waters} waters for MM/GBSA-NWAT but only {len(indices.waters)} are available."
        raise ValueError(errmsg)
    water_slots = [i for w in indices.waters[:n_waters] for i in w]
    receptor = indices.receptor + water_slots
    return {
        "complex": receptor + indices.ligand,
        "receptor": receptor,
        "ligand": list(indices.ligand),
    }


def create_end_state_systems(
    system: openmm.System,
    topology: app.Topology,
    indices: MMGBSAAtomIndices,
    settings: MMGBSAProtocolSettings,
) -> dict[str, openmm.System]:
    """
    Create the generalized Born Systems for each MM/GBSA end state.

    Parameters
    ----------
    system : openmm.System
      The explicit solvent System.
    topology : openmm.app.Topology
      The explicit solvent System's Topology.
    indices : MMGBSAAtomIndices
      The MM/GBSA atom indices.
    settings : MMGBSAProtocolSettings
      The Protocol settings.

    Returns
    -------
    dict[str, openmm.System]
      The complex, receptor and ligand generalized Born Systems.
    """
    gb_settings = settings.mmgbsa_settings
    kappa = get_debye_kappa(
        gb_settings.salt_concentration,
        gb_settings.solvent_dielectric,
        settings.thermo_settings.temperature,
    )

    systems = {}
    for state, atoms in get_end_state_atoms(indices, gb_settings.n_waters).items():
        systems[state], _ = create_gb_system(
            system=system,
            topology=topology,
            atom_indices=atoms,
            gb_model=gb_settings.gb_model,
            solute_dielectric=gb_settings.solute_dielectric,
            solvent_dielectric=gb_settings.solvent_dielectric,
            kappa=kappa,
            surface_tension=gb_settings.surface_tension,
            probe_radius=gb_settings.probe_radius,
        )
    return systems


def _get_ligand(stateA: ChemicalSystem) -> gufe.SmallMoleculeComponent:
    _, _, small_mols = system_validation.get_components(stateA)
    return small_mols[0]


def _get_protein(stateA: ChemicalSystem) -> gufe.ProteinComponent:
    _, protein, _ = system_validation.get_components(stateA)
    return protein


class MMGBSASetupUnit(PlainMDSetupUnit):
    """
    Setup unit for the MM/GBSA Protocol.

    Extends the PlainMD setup unit, additionally identifying the receptor,
    ligand and water atoms and creating the generalized Born Systems
    for each end state.
    """

    def run(
        self,
        *,
        dry: bool = False,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Setup the MM/GBSA system.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary
          components but without serializing the end state systems.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        scratch_basepath: pathlib.Path | None
          Where to store temporary files, defaults to current working directory
        shared_basepath : pathlib.Path | None
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          Outputs created by the setup unit or the debug objects
          if ``dry==True``.
        """
        outputs = super().run(
            dry=dry,
            verbose=verbose,
            scratch_basepath=scratch_basepath,
            shared_basepath=shared_basepath,
        )

        settings: MMGBSAProtocolSettings = self._inputs["protocol"].settings
        stateA = self._inputs["stateA"]
        n_waters = settings.mmgbsa_settings.n_waters

        if dry:
            system = outputs["debug"]["system"]
        else:
            system = serialization.deserialize(outputs["system"])

        # Use the same Topology as the simulation unit, i.e. the one read
        # back from the PDB file
        topology = app.PDBFile(str(outputs["system_pdb"])).getTopology()

        if verbose:
            self.logger.info("Identifying MM/GBSA receptor, ligand and water atoms")

        indices = get_mmgbsa_atom_indices(
            topology=topology,
            protein_component=_get_protein(stateA),
            ligand_component=_get_ligand(stateA),
            output_selection=settings.output_settings.output_indices,
        )

        if n_waters > 0:
            check_waters_in_trajectory(indices, settings.output_settings.output_indices)
            check_identical_water_parameters(system, indices.waters)

        if verbose:
            self.logger.info("Creating MM/GBSA end state systems")

        gb_systems = create_end_state_systems(system, topology, indices, settings)

        indices_file = self.shared_basepath / "mmgbsa_atom_indices.json"
        indices.to_json(indices_file)
        outputs["mmgbsa_atom_indices"] = indices_file

        if dry:
            outputs["debug"]["mmgbsa_atom_indices"] = indices
            outputs["debug"]["gb_systems"] = gb_systems
            outputs["debug"]["topology_from_pdb"] = topology
        else:
            for state, gb_system in gb_systems.items():
                outfile = self.shared_basepath / f"gb_{state}_system.xml.bz2"
                serialization.serialize(gb_system, outfile)
                outputs[f"gb_{state}_system"] = outfile

        return outputs


class MMGBSASimulationUnit(PlainMDSimulationUnit):
    """
    Simulation unit for the MM/GBSA Protocol.

    This is the PlainMD simulation unit, it runs minimization,
    equilibration and the production simulation.
    """


class MMGBSAAnalysisUnit(PlainMDUnitMixin, gufe.ProtocolUnit):
    """
    Analysis unit for the MM/GBSA Protocol.

    Re-scores the production trajectory using single trajectory MM/GBSA,
    optionally retaining the closest waters to the ligand (MM/GBSA-NWAT).
    """

    @staticmethod
    def _get_start_frame(settings: MMGBSAProtocolSettings) -> int:
        """
        Get the first trajectory frame to analyze.

        The trajectory reporter does not write the initial production
        frame, so frame ``i`` is at production time ``(i + 1) * interval``.
        """
        start_time = settings.mmgbsa_settings.analysis_start_time
        if start_time is None:
            return 0
        interval = settings.output_settings.trajectory_write_interval
        n_intervals = int(math.ceil((start_time / interval).to("dimensionless").m - 1e-8))
        return max(n_intervals - 1, 0)

    def run(
        self,
        *,
        gb_systems: dict[str, openmm.System],
        topology: app.Topology,
        indices: MMGBSAAtomIndices,
        trajectory: pathlib.Path,
        verbose: bool = True,
        scratch_basepath: pathlib.Path | None = None,
        shared_basepath: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        """Re-score the production trajectory with MM/GBSA.

        Parameters
        ----------
        gb_systems : dict[str, openmm.System]
          The generalized Born Systems for each end state.
        topology : openmm.app.Topology
          The Topology of the full explicit solvent System.
        indices : MMGBSAAtomIndices
          The MM/GBSA atom indices.
        trajectory : pathlib.Path
          The production trajectory.
        verbose : bool
          Verbose output. Output is provided via INFO level logging.
        scratch_basepath: pathlib.Path | None
          Where to store temporary files, defaults to current working directory
        shared_basepath : pathlib.Path | None
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          The MM/GBSA summary and the path to the per-frame energies.
        """
        self._prepare(verbose, scratch_basepath, shared_basepath)

        settings: MMGBSAProtocolSettings = self._inputs["protocol"].settings
        gb_settings = settings.mmgbsa_settings

        ligand_offmol = _get_ligand(self._inputs["stateA"]).to_openff()
        if gb_settings.water_selection == "heavy":
            ligand_mask = np.array([a.atomic_number > 1 for a in ligand_offmol.atoms])
        else:
            ligand_mask = np.ones(ligand_offmol.n_atoms, dtype=bool)

        platform = omm_compute.get_openmm_platform(
            platform_name=settings.engine_settings.compute_platform,
            gpu_device_index=settings.engine_settings.gpu_device_index,
            restrict_cpu_count=False,
        )

        universe = get_universe(topology, indices, trajectory)
        start = self._get_start_frame(settings)
        if start >= universe.trajectory.n_frames:
            errmsg = (
                f"analysis_start_time leaves no frames to analyze, the trajectory "
                f"has {universe.trajectory.n_frames} frames."
            )
            raise ValueError(errmsg)

        frames = iterate_frames(
            universe=universe,
            indices=indices,
            ligand_selection_mask=ligand_mask,
            n_waters=gb_settings.n_waters,
            start=start,
            stride=gb_settings.frame_stride,
        )

        if self.verbose:
            self.logger.info("Scoring frames with MM/GBSA")

        evaluator = EndStateEvaluator(gb_systems, platform)
        try:
            rows = score_frames(evaluator, frames)
        finally:
            evaluator.close()

        energies_file = self.shared_basepath / "mmgbsa_energies.csv"
        write_rows(rows, energies_file)
        summary = summarize(rows)

        if self.verbose:
            total = summary["total"]
            self.logger.info(
                f"MM/GBSA binding energy: {total['mean']:.2f} +/- {total['sem']:.2f} kcal/mol"
            )

        return {
            "per_frame_energies": energies_file,
            "summary": summary,
        }

    def _execute(
        self,
        ctx: gufe.Context,
        setup_results,
        simulation_results,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        gb_systems = {
            state: serialization.deserialize(setup_results.outputs[f"gb_{state}_system"])
            for state in STATES
        }
        topology = app.PDBFile(str(setup_results.outputs["system_pdb"])).getTopology()
        indices = MMGBSAAtomIndices.from_json(setup_results.outputs["mmgbsa_atom_indices"])

        outputs = self.run(
            gb_systems=gb_systems,
            topology=topology,
            indices=indices,
            trajectory=simulation_results.outputs["nc"],
            scratch_basepath=ctx.scratch,
            shared_basepath=ctx.shared,
        )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }

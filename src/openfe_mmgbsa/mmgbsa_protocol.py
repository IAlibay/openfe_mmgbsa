"""OpenMM MM/GBSA Protocol.

Runs explicit solvent MD of a protein-ligand complex using the OpenFE
PlainMD machinery and then re-scores the production trajectory using
single trajectory MM/GBSA, with all energies evaluated with OpenMM.

Note
----
MM/GBSA binding energies are effective binding scores; no configurational
entropy is included and they should not be interpreted as rigorous binding
free energies.
"""

import pathlib
import uuid
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional

import gufe
import mdtraj
import numpy as np
from gufe import (
    BaseSolventComponent,
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
)
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocol
from openff.units import Quantity, unit

from .mmgbsa_settings import MMGBSAProtocolSettings, MMGBSASettings
from .mmgbsa_units import MMGBSAAnalysisUnit, MMGBSASetupUnit, MMGBSASimulationUnit
from .scoring import read_rows


class MMGBSAProtocolResult(gufe.ProtocolResult):
    """
    Dict-like container for the output of an MMGBSAProtocol.

    All energies are in kcal/mol.
    """

    def __init__(self, **data):
        super().__init__(**data)
        # data is mapping of str(repeat_id): list[protocolunitresults]
        if any(len(pur_list) > 1 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    def _summaries(self) -> list[dict]:
        return [pus[0].outputs["summary"] for pus in self.data.values()]

    def get_individual_estimates(self) -> list[tuple[Quantity, Quantity]]:
        """
        Get the MM/GBSA binding energy and its correlation corrected
        standard error of the mean for each repeat.

        Returns
        -------
        list[tuple[openff.units.Quantity, openff.units.Quantity]]
          The (estimate, uncertainty) of each repeat.
        """
        return [
            (s["total"]["mean"] * unit.kilocalorie_per_mole, s["total"]["sem"] * unit.kilocalorie_per_mole)
            for s in self._summaries()
        ]

    def get_estimate(self) -> Quantity:
        """
        Get the MM/GBSA binding energy, averaged over all repeats.

        Returns
        -------
        openff.units.Quantity
          The MM/GBSA binding energy.
        """
        means = [s["total"]["mean"] for s in self._summaries()]
        return float(np.mean(means)) * unit.kilocalorie_per_mole

    def get_uncertainty(self) -> Quantity:
        """
        Get the uncertainty of the MM/GBSA binding energy.

        This is the standard deviation of the repeat estimates. If there is
        only a single repeat, the correlation corrected standard error of
        the mean of that repeat is returned instead.

        Returns
        -------
        openff.units.Quantity
          The uncertainty of the MM/GBSA binding energy.
        """
        summaries = self._summaries()
        if len(summaries) == 1:
            return summaries[0]["total"]["sem"] * unit.kilocalorie_per_mole
        means = [s["total"]["mean"] for s in summaries]
        return float(np.std(means, ddof=1)) * unit.kilocalorie_per_mole

    def get_components(self) -> dict[str, tuple[Quantity, Quantity]]:
        """
        Get the decomposed MM/GBSA binding energy, averaged over all repeats.

        The components are: ``bond``, ``angle``, ``torsion`` (all of which
        should be zero for single trajectory MM/GBSA), ``elec``, ``vdw``,
        ``gb_polar``, ``gb_nonpolar`` and the sums ``bonded``, ``mm``,
        ``solvation`` and ``total``.

        Returns
        -------
        dict[str, tuple[openff.units.Quantity, openff.units.Quantity]]
          The (estimate, uncertainty) of each component. The uncertainty
          follows the same convention as :meth:`get_uncertainty`.
        """
        summaries = self._summaries()
        components = {}
        for comp in summaries[0]:
            means = [s[comp]["mean"] for s in summaries]
            if len(summaries) == 1:
                err = summaries[0][comp]["sem"]
            else:
                err = float(np.std(means, ddof=1))
            components[comp] = (
                float(np.mean(means)) * unit.kilocalorie_per_mole,
                err * unit.kilocalorie_per_mole,
            )
        return components

    def get_per_frame_energies_filenames(self) -> list[pathlib.Path]:
        """
        Get the paths to the per-frame energy CSV file of each repeat.
        """
        return [pus[0].outputs["per_frame_energies"] for pus in self.data.values()]

    def get_per_frame_energies(self) -> list[list[dict]]:
        """
        Get the per-frame energies of each repeat.

        Returns
        -------
        list[list[dict]]
          For each repeat, a list of rows (one per frame) with keys
          ``frame``, ``water_ids`` and ``{state}_{component}`` where state
          is one of complex, receptor, ligand or delta.
        """
        return [read_rows(f) for f in self.get_per_frame_energies_filenames()]


class MMGBSAProtocol(PlainMDProtocol):
    """
    Protocol for single trajectory MM/GBSA, optionally retaining a number of
    explicit waters closest to the ligand (MM/GBSA-NWAT).

    Explicit solvent MD is run using the same approach as
    :class:`openfe.protocols.openmm_md.PlainMDProtocol`, and frames of the
    production trajectory are re-scored using OpenMM generalized Born
    implicit solvent models.

    See Also
    --------
    :class:`openfe_mmgbsa.MMGBSAProtocolSettings`
    :class:`openfe_mmgbsa.MMGBSAProtocolResult`
    """

    result_cls = MMGBSAProtocolResult
    _settings_cls = MMGBSAProtocolSettings
    _settings: MMGBSAProtocolSettings

    @classmethod
    def _default_settings(cls):
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        Settings
          a set of default settings
        """
        md_settings = PlainMDProtocol._default_settings()
        settings = MMGBSAProtocolSettings(
            **{field: getattr(md_settings, field) for field in type(md_settings).model_fields},
            mmgbsa_settings=MMGBSASettings(),
        )
        # Only the solute is needed for standard MM/GBSA
        settings.output_settings.output_indices = "not water"
        return settings

    @staticmethod
    def _selection_includes_water(selection: str) -> bool:
        """Check if an MDTraj selection string selects water."""
        top = mdtraj.Topology()
        chain = top.add_chain()
        res = top.add_residue("HOH", chain)
        oxygen = top.add_atom("O", mdtraj.element.oxygen, res)
        top.add_atom("H1", mdtraj.element.hydrogen, res)
        top.add_atom("H2", mdtraj.element.hydrogen, res)
        return oxygen.index in top.select(selection)

    def _validate(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, gufe.ComponentMapping]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ):
        # MM/GBSA specific checks first, since they give clearer errors
        if len(stateA.get_components_of_type(ProteinComponent)) != 1:
            raise ValueError("MM/GBSA requires exactly one ProteinComponent (the receptor).")

        if len(stateA.get_components_of_type(SmallMoleculeComponent)) != 1:
            raise ValueError("MM/GBSA requires exactly one SmallMoleculeComponent (the ligand).")

        if not stateA.contains(BaseSolventComponent):
            raise ValueError("MM/GBSA requires an explicitly solvated system.")

        if self.settings.forcefield_settings.nonbonded_method.lower() == "nocutoff":
            raise ValueError("MM/GBSA requires a periodic nonbonded method (e.g. PME).")

        if self.settings.mmgbsa_settings.n_waters > 0 and not self._selection_includes_water(
            self.settings.output_settings.output_indices
        ):
            errmsg = (
                "n_waters > 0 (MM/GBSA-NWAT) requires the waters to be written to "
                "the trajectory, but output_settings.output_indices "
                f"('{self.settings.output_settings.output_indices}') does not select "
                "water. Consider using output_indices='all'."
            )
            raise ValueError(errmsg)

        # Generic PlainMD checks
        super()._validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, gufe.ComponentMapping]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # validate the inputs
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        ligand = stateA.get_components_of_type(SmallMoleculeComponent)[0]
        protein = stateA.get_components_of_type(ProteinComponent)[0]
        system_name = f"{protein.name or 'NoName'}:{ligand.name or 'NoName'}"

        units = []
        for i in range(self.settings.protocol_repeats):
            repeat_id = int(uuid.uuid4())

            setup = MMGBSASetupUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=repeat_id,
                name=f"MMGBSA Setup: {system_name} repeat {i} generation 0",
            )
            sim = MMGBSASimulationUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=repeat_id,
                setup_results=setup,
                name=f"MMGBSA Simulation: {system_name} repeat {i} generation 0",
            )
            analysis = MMGBSAAnalysisUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=repeat_id,
                setup_results=setup,
                simulation_results=sim,
                name=f"MMGBSA Analysis: {system_name} repeat {i} generation 0",
            )

            units.extend([setup, sim, analysis])

        return units

    def _gather(self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]) -> dict[str, Any]:
        # group the successful analysis unit results by repeat_id
        unsorted_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if (not pu.ok()) or ("summary" not in pu.outputs):
                    continue
                unsorted_repeats[pu.outputs["repeat_id"]].append(pu)

        # then sort by generation within each repeat_id list
        repeats: dict[str, list[gufe.ProtocolUnitResult]] = {}
        for k, v in unsorted_repeats.items():
            repeats[str(k)] = sorted(v, key=lambda x: x.outputs["generation"])

        return repeats

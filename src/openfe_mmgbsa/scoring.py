"""Per-frame MM/GBSA energy evaluation and statistics."""

import csv
import pathlib
import warnings
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import openmm
from openmm import unit as omm_unit

from .frames import MMGBSAFrame
from .gb_systems import FORCE_GROUPS

KJ_TO_KCAL = 1 / 4.184

ENERGY_COMPONENTS = list(FORCE_GROUPS)
"""The individual energy components, each in their own force group."""

MM_COMPONENTS = ["bond", "angle", "torsion", "elec", "vdw"]
BONDED_COMPONENTS = ["bond", "angle", "torsion"]
SOLVATION_COMPONENTS = ["gb_polar", "gb_nonpolar"]

DERIVED_COMPONENTS = ["bonded", "mm", "solvation", "total"]
"""Sums of the energy components which are also reported."""

STATES = ["complex", "receptor", "ligand"]


def _add_derived(energies: dict[str, float]) -> dict[str, float]:
    energies["bonded"] = sum(energies[c] for c in BONDED_COMPONENTS)
    energies["mm"] = sum(energies[c] for c in MM_COMPONENTS)
    energies["solvation"] = sum(energies[c] for c in SOLVATION_COMPONENTS)
    energies["total"] = energies["mm"] + energies["solvation"]
    return energies


class EndStateEvaluator:
    """
    Evaluates the decomposed potential energy of the complex, receptor and
    ligand generalized Born Systems.

    The complex System must have its atoms ordered as receptor, NWAT waters
    and then ligand; the receptor System as receptor then NWAT waters.
    """

    def __init__(
        self,
        systems: dict[str, openmm.System],
        platform: openmm.Platform,
    ):
        self.contexts = {}
        # Keep a reference to the integrators so they are not garbage collected
        self._integrators = {}
        for state in STATES:
            integrator = openmm.VerletIntegrator(1.0 * omm_unit.femtoseconds)
            self._integrators[state] = integrator
            self.contexts[state] = openmm.Context(systems[state], integrator, platform)

    def _energies(self, state: str, positions: npt.NDArray) -> dict[str, float]:
        """Get the energy components in kcal/mol."""
        context = self.contexts[state]
        context.setPositions(positions * omm_unit.nanometer)
        energies = {}
        for name, group in FORCE_GROUPS.items():
            e = context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
            energies[name] = e.value_in_unit(omm_unit.kilojoule_per_mole) * KJ_TO_KCAL
        return _add_derived(energies)

    def evaluate(self, frame: MMGBSAFrame) -> dict[str, dict[str, float]]:
        """
        Get the energy components for each state and the binding energy
        (``delta``) for a frame.

        Returns
        -------
        dict[str, dict[str, float]]
          Energies in kcal/mol, keyed by state (complex, receptor, ligand
          and delta) and then energy component.
        """
        receptor = np.concatenate([frame.receptor, frame.waters])
        complex_ = np.concatenate([receptor, frame.ligand])
        energies = {
            "complex": self._energies("complex", complex_),
            "receptor": self._energies("receptor", receptor),
            "ligand": self._energies("ligand", frame.ligand),
        }
        energies["delta"] = {
            c: energies["complex"][c] - energies["receptor"][c] - energies["ligand"][c]
            for c in energies["complex"]
        }
        return energies

    def close(self) -> None:
        del self.contexts, self._integrators


def score_frames(
    evaluator: EndStateEvaluator,
    frames: Iterable[MMGBSAFrame],
    bonded_tolerance: float = 1e-2,
) -> list[dict]:
    """
    Evaluate all the frames.

    Parameters
    ----------
    evaluator : EndStateEvaluator
      The end state evaluator.
    frames : Iterable[MMGBSAFrame]
      The frames to evaluate.
    bonded_tolerance : float
      Tolerance, in kcal/mol, for the bonded binding energy contribution
      to differ from zero before a warning is issued. In the single
      trajectory approach the bonded terms should exactly cancel.

    Returns
    -------
    list[dict]
      One row per frame, with keys ``frame``, ``water_ids`` and
      ``{state}_{component}`` for each state and energy component.
    """
    rows = []
    warned = False
    for frame in frames:
        energies = evaluator.evaluate(frame)
        row = {"frame": frame.frame, "water_ids": frame.water_ids}
        for state, values in energies.items():
            for comp, value in values.items():
                row[f"{state}_{comp}"] = value
        if abs(row["delta_bonded"]) > bonded_tolerance and not warned:
            wmsg = (
                f"Bonded binding energy contribution of {row['delta_bonded']:.4f} kcal/mol "
                f"found for frame {frame.frame}. In single trajectory MM/GBSA this "
                "should be zero, this may indicate an imaging problem or that "
                "the receptor and ligand are covalently bonded."
            )
            warnings.warn(wmsg)
            warned = True
        rows.append(row)
    return rows


def write_rows(rows: list[dict], filename: pathlib.Path) -> None:
    """Write the per-frame energies to a CSV file."""
    if not rows:
        raise ValueError("No frames to write")
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["water_ids"] = " ".join(str(i) for i in row["water_ids"])
            writer.writerow(out)


def read_rows(filename: pathlib.Path) -> list[dict]:
    """Read the per-frame energies written by :func:`write_rows`."""
    rows = []
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            out = {}
            for k, v in row.items():
                if k == "frame":
                    out[k] = int(v)
                elif k == "water_ids":
                    out[k] = [int(i) for i in v.split()]
                else:
                    out[k] = float(v)
            rows.append(out)
    return rows


def get_statistical_inefficiency(series: npt.NDArray) -> float:
    """
    Get the statistical inefficiency of a timeseries, returning 1.0 for
    series which are too short or have no variance.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 3 or np.allclose(series, series[0]):
        return 1.0

    from pymbar import timeseries

    return float(max(timeseries.statistical_inefficiency(series), 1.0))


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    """
    Get the mean, and the standard error of the mean corrected for
    correlation, of each binding energy component.

    Returns
    -------
    dict[str, dict[str, float]]
      Keyed by energy component, values are a dict with keys ``mean``,
      ``sem``, ``std``, ``g`` (statistical inefficiency) and ``n_frames``.
      Energies are in kcal/mol.
    """
    summary = {}
    for comp in ENERGY_COMPONENTS + DERIVED_COMPONENTS:
        series = np.array([r[f"delta_{comp}"] for r in rows])
        n = len(series)
        std = float(series.std(ddof=1)) if n > 1 else 0.0
        g = get_statistical_inefficiency(series)
        summary[comp] = {
            "mean": float(series.mean()),
            "std": std,
            "sem": float(std * np.sqrt(g / n)) if n > 0 else 0.0,
            "g": g,
            "n_frames": n,
        }
    return summary

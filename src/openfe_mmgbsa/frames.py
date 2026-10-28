"""Reading, imaging and NWAT water selection of trajectory frames using MDAnalysis."""

import pathlib
from collections.abc import Iterator
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from MDAnalysis import transformations as trans
from MDAnalysis.lib.distances import distance_array, minimize_vectors
from openmm import app

from .atom_indices import MMGBSAAtomIndices
from .gb_systems import subset_topology


@dataclass
class MMGBSAFrame:
    """The coordinates of one frame, in nanometers."""

    frame: int
    """The index of the frame in the trajectory."""

    receptor: npt.NDArray
    """The receptor positions, shape (n_receptor_atoms, 3)."""

    ligand: npt.NDArray
    """The ligand positions, shape (n_ligand_atoms, 3)."""

    waters: npt.NDArray
    """The NWAT water positions, shape (n_waters * 3, 3)."""

    water_ids: list[int]
    """The index (in ``MMGBSAAtomIndices.waters``) of each retained water."""


def get_universe(
    topology: app.Topology,
    indices: MMGBSAAtomIndices,
    trajectory: pathlib.Path,
) -> mda.Universe:
    """
    Create an MDAnalysis Universe for the production trajectory.

    Parameters
    ----------
    topology : openmm.app.Topology
      The Topology of the full System.
    indices : MMGBSAAtomIndices
      The MM/GBSA atom indices, including those written to the trajectory.
    trajectory : pathlib.Path
      The production trajectory file.

    Returns
    -------
    MDAnalysis.Universe
      The Universe; its atoms correspond to ``indices.trajectory``.
    """
    traj_top = subset_topology(topology, indices.trajectory)
    return mda.Universe(traj_top, str(trajectory))


def _to_traj_indices(full_indices, trajectory_indices: npt.NDArray) -> npt.NDArray:
    """Convert full System atom indices to trajectory atom indices."""
    full_indices = np.asarray(full_indices, dtype=int)
    traj_idx = np.searchsorted(trajectory_indices, full_indices)
    if np.any(traj_idx >= len(trajectory_indices)) or np.any(
        trajectory_indices[np.minimum(traj_idx, len(trajectory_indices) - 1)] != full_indices
    ):
        raise ValueError("Requested atoms are not present in the trajectory")
    return traj_idx


def select_closest_waters(
    ligand_positions: npt.NDArray,
    oxygen_positions: npt.NDArray,
    n_waters: int,
    box: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Find the waters closest to the ligand using minimum image distances.

    Parameters
    ----------
    ligand_positions : npt.NDArray
      The positions of the ligand atoms used for the selection.
    oxygen_positions : npt.NDArray
      The positions of every water oxygen.
    n_waters : int
      The number of waters to select.
    box : npt.NDArray
      The MDAnalysis box dimensions.

    Returns
    -------
    selected : npt.NDArray
      The indices of the selected waters, ordered by increasing distance.
    closest_ligand_atom : npt.NDArray
      For each selected water, the index of the ligand atom closest to it.
    """
    if n_waters > len(oxygen_positions):
        errmsg = f"Requested {n_waters} waters but only {len(oxygen_positions)} are available"
        raise ValueError(errmsg)

    distances = distance_array(ligand_positions, oxygen_positions, box=box)
    min_distances = distances.min(axis=0)
    # stable sort so that ties are broken deterministically
    selected = np.argsort(min_distances, kind="stable")[:n_waters]
    closest_ligand_atom = distances[:, selected].argmin(axis=0)
    return selected, closest_ligand_atom


def iterate_frames(
    universe: mda.Universe,
    indices: MMGBSAAtomIndices,
    ligand_selection_mask: npt.NDArray,
    n_waters: int,
    start: int = 0,
    stride: int = 1,
) -> Iterator[MMGBSAFrame]:
    """
    Iterate over imaged trajectory frames.

    Each frame is imaged such that the receptor and ligand molecules are whole
    and in the same periodic image, with the receptor centered in the box.
    If ``n_waters > 0``, the ``n_waters`` waters closest to the ligand are
    also extracted, with each water placed in the periodic image closest to
    the ligand.

    Parameters
    ----------
    universe : MDAnalysis.Universe
      A Universe created with :func:`get_universe`.
    indices : MMGBSAAtomIndices
      The MM/GBSA atom indices.
    ligand_selection_mask : npt.NDArray
      A boolean mask over the ligand atoms defining which atoms are used
      to find the closest waters.
    n_waters : int
      The number of closest waters to extract.
    start : int
      The first frame to use. Default 0.
    stride : int
      The frame stride. Default 1.

    Yields
    ------
    MMGBSAFrame
      The coordinates for the frame, in nanometers.
    """
    trajectory_indices = np.asarray(indices.trajectory, dtype=int)
    receptor = universe.atoms[_to_traj_indices(indices.receptor, trajectory_indices)]
    ligand = universe.atoms[_to_traj_indices(indices.ligand, trajectory_indices)]
    solute = receptor + ligand

    if n_waters > 0:
        water_atoms = universe.atoms[
            _to_traj_indices(np.ravel(indices.waters), trajectory_indices)
        ]
        oxygens = water_atoms[::3]
    else:
        water_atoms = oxygens = None

    # OpenMM wraps molecules into the box, make sure that the receptor and
    # ligand are whole and in the same periodic image
    universe.trajectory.add_transformations(
        trans.unwrap(solute),
        trans.center_in_box(receptor, center="geometry"),
        trans.wrap(solute, compound="fragments"),
    )

    for ts in universe.trajectory[start::stride]:
        box = ts.dimensions
        ligand_pos = ligand.positions

        if n_waters > 0:
            selected, closest = select_closest_waters(
                ligand_pos[ligand_selection_mask],
                oxygens.positions,
                n_waters,
                box,
            )
            water_pos = water_atoms.positions.reshape(-1, 3, 3)[selected]
            # shift each water so its oxygen is in the image closest to the ligand
            ref = ligand_pos[ligand_selection_mask][closest]
            vec = water_pos[:, 0, :] - ref
            shift = minimize_vectors(vec, box) - vec
            water_pos = (water_pos + shift[:, None, :]).reshape(-1, 3)
            water_ids = [int(i) for i in selected]
        else:
            water_pos = np.zeros((0, 3))
            water_ids = []

        # MDAnalysis uses Angstrom, OpenMM nanometers
        yield MMGBSAFrame(
            frame=ts.frame,
            receptor=receptor.positions / 10.0,
            ligand=ligand_pos / 10.0,
            waters=water_pos / 10.0,
            water_ids=water_ids,
        )

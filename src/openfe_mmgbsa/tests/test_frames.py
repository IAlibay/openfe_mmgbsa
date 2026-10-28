import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader
from numpy.testing import assert_allclose
from openmm import app

from openfe_mmgbsa.atom_indices import MMGBSAAtomIndices
from openfe_mmgbsa.frames import _to_traj_indices, iterate_frames, select_closest_waters
from openfe_mmgbsa.gb_systems import subset_topology

BOX = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0])


def test_select_closest_waters_pbc():
    ligand = np.array([[1.0, 15.0, 15.0], [2.0, 15.0, 15.0]])
    oxygens = np.array(
        [
            [15.0, 15.0, 15.0],  # 13 A
            [28.0, 15.0, 15.0],  # 3 A through the boundary from ligand atom 0
            [6.0, 15.0, 15.0],  # 4 A from ligand atom 1
        ]
    )
    selected, closest = select_closest_waters(ligand, oxygens, 2, BOX)
    assert list(selected) == [1, 2]
    assert list(closest) == [0, 1]


def test_select_too_many_waters():
    with pytest.raises(ValueError, match="only 1 are available"):
        select_closest_waters(np.zeros((1, 3)), np.zeros((1, 3)), 2, BOX)


def test_to_traj_indices():
    traj = np.array([0, 2, 5, 7])
    assert list(_to_traj_indices([2, 7], traj)) == [1, 3]
    for bad in ([3], [8]):
        with pytest.raises(ValueError, match="not present"):
            _to_traj_indices(bad, traj)


@pytest.fixture
def toy_system():
    """
    A 2-atom receptor, a 1-atom ligand and 3 waters in a 3 nm cubic box,
    with the receptor and ligand split across the periodic boundary.
    """
    top = app.Topology()
    chain = top.addChain()
    rec = top.addResidue("REC", chain)
    c1 = top.addAtom("C1", app.element.carbon, rec)
    c2 = top.addAtom("C2", app.element.carbon, rec)
    top.addBond(c1, c2)
    lig = top.addResidue("LIG", chain)
    top.addAtom("C", app.element.carbon, lig)
    for _ in range(3):
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", app.element.oxygen, res)
        for name in ["H1", "H2"]:
            top.addBond(o, top.addAtom(name, app.element.hydrogen, res))

    positions = np.array(
        [
            [0.5, 15.0, 15.0],  # receptor C1
            [28.5, 15.0, 15.0],  # receptor C2, bonded to C1 through the boundary
            [26.5, 15.0, 15.0],  # ligand
            [3.5, 15.0, 15.0],  # water 0, 7 A from the ligand through the boundary
            [4.5, 15.0, 15.0],
            [3.5, 16.0, 15.0],
            [22.0, 15.0, 15.0],  # water 1, 4.5 A from the ligand
            [21.0, 15.0, 15.0],
            [22.0, 16.0, 15.0],
            [12.0, 15.0, 15.0],  # water 2, far away
            [11.0, 15.0, 15.0],
            [12.0, 16.0, 15.0],
        ]
    )
    indices = MMGBSAAtomIndices(
        receptor=[0, 1],
        ligand=[2],
        waters=[[3, 4, 5], [6, 7, 8], [9, 10, 11]],
        trajectory=list(range(12)),
    )
    universe = mda.Universe(
        subset_topology(top, indices.trajectory),
        positions[None, :, :],
        format=MemoryReader,
        dimensions=BOX,
    )
    return universe, indices, positions


def test_iterate_frames_imaging(toy_system):
    universe, indices, _ = toy_system
    frames = list(iterate_frames(universe, indices, np.array([True]), n_waters=2))
    assert len(frames) == 1
    frame = frames[0]
    assert frame.frame == 0
    assert frame.water_ids == [1, 0]

    # nanometers, with the receptor whole and the ligand in the same image
    assert frame.receptor.shape == (2, 3)
    assert np.linalg.norm(frame.receptor[0] - frame.receptor[1]) == pytest.approx(0.2, abs=1e-5)
    receptor_center = frame.receptor.mean(axis=0)
    assert_allclose(receptor_center, BOX[:3] / 20.0, atol=1e-5)
    assert np.linalg.norm(frame.ligand[0] - frame.receptor[1]) == pytest.approx(0.2, abs=1e-5)

    # waters are whole and in the image closest to the ligand
    waters = frame.waters.reshape(-1, 3, 3)
    assert waters.shape == (2, 3, 3)
    for water, expected in zip(waters, [0.45, 0.7]):
        assert np.linalg.norm(water[0] - frame.ligand[0]) == pytest.approx(expected, abs=1e-5)
        assert_allclose(np.linalg.norm(water[1:] - water[0], axis=1), [0.1, 0.1], atol=1e-5)


def test_iterate_frames_no_waters(toy_system):
    universe, indices, _ = toy_system
    frame = next(iterate_frames(universe, indices, np.array([True]), n_waters=0))
    assert frame.waters.shape == (0, 3)
    assert frame.water_ids == []

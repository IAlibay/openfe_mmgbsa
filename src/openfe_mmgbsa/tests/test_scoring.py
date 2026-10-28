import numpy as np
import pytest
from numpy.testing import assert_allclose

from openfe_mmgbsa.frames import MMGBSAFrame
from openfe_mmgbsa.scoring import (
    DERIVED_COMPONENTS,
    ENERGY_COMPONENTS,
    get_statistical_inefficiency,
    read_rows,
    score_frames,
    summarize,
    write_rows,
)


def _row(frame, total, water_ids=(), bonded=0.0):
    row = {"frame": frame, "water_ids": list(water_ids)}
    for state in ["complex", "receptor", "ligand", "delta"]:
        for comp in ENERGY_COMPONENTS + DERIVED_COMPONENTS:
            row[f"{state}_{comp}"] = 0.0
    row["delta_total"] = total
    row["delta_bonded"] = bonded
    return row


def test_statistical_inefficiency_edge_cases():
    assert get_statistical_inefficiency([1.0, 2.0]) == 1.0
    assert get_statistical_inefficiency(np.ones(10)) == 1.0
    # strongly correlated series
    series = np.repeat(np.random.default_rng(0).normal(size=20), 10)
    assert get_statistical_inefficiency(series) > 5


def test_summarize():
    totals = [-10.0, -12.0, -11.0, -13.0]
    summary = summarize([_row(i, t) for i, t in enumerate(totals)])
    assert set(summary) == set(ENERGY_COMPONENTS + DERIVED_COMPONENTS)
    total = summary["total"]
    assert total["mean"] == pytest.approx(-11.5)
    assert total["std"] == pytest.approx(np.std(totals, ddof=1))
    assert total["n_frames"] == 4
    assert total["sem"] == pytest.approx(total["std"] * np.sqrt(total["g"] / 4))
    assert summary["elec"] == {"mean": 0.0, "std": 0.0, "sem": 0.0, "g": 1.0, "n_frames": 4}


def test_summarize_single_frame():
    summary = summarize([_row(0, -5.0)])
    assert summary["total"]["mean"] == -5.0
    assert summary["total"]["sem"] == 0.0


def test_rows_roundtrip(tmp_path):
    rows = [_row(0, -1.5, [3, 1]), _row(5, -2.25, [])]
    fname = tmp_path / "energies.csv"
    write_rows(rows, fname)
    assert read_rows(fname) == rows


def test_write_no_rows(tmp_path):
    with pytest.raises(ValueError, match="No frames"):
        write_rows([], tmp_path / "energies.csv")


class _FakeEvaluator:
    def __init__(self, bonded):
        self.bonded = bonded

    def evaluate(self, frame):
        comps = ENERGY_COMPONENTS + DERIVED_COMPONENTS
        energies = {s: {c: 0.0 for c in comps} for s in ["complex", "receptor", "ligand"]}
        energies["delta"] = {c: 0.0 for c in comps}
        energies["delta"]["bonded"] = self.bonded
        energies["delta"]["total"] = -float(frame.frame)
        return energies


def _frames(n):
    empty = np.zeros((0, 3))
    for i in range(n):
        yield MMGBSAFrame(frame=i, receptor=empty, ligand=empty, waters=empty, water_ids=[i])


def test_score_frames():
    rows = score_frames(_FakeEvaluator(0.0), _frames(3))
    assert [r["frame"] for r in rows] == [0, 1, 2]
    assert [r["water_ids"] for r in rows] == [[0], [1], [2]]
    assert_allclose([r["delta_total"] for r in rows], [0, -1, -2])


def test_score_frames_bonded_warning():
    with pytest.warns(UserWarning, match="Bonded binding energy"):
        rows = score_frames(_FakeEvaluator(1.0), _frames(3))
    assert len(rows) == 3

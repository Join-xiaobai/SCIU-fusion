"""Block-NEES check for SCIU (Path B). CPU only.

CI/CU theorems are *conditional* consistency of the reported ellipsoid,
not Loewner of averaged covariances. This test therefore checks overlap
and complement NEES, and records that joint Loewner of E[P]-E[ee^T] is
not claimed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sciu_2x2_pilot import CELLS, fuse_cell, simulate_cell  # noqa: E402


def _block_nees(err, cov):
    err = np.asarray(err, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    try:
        sol = np.linalg.solve(cov, err[:, :, None])
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(cov) @ err[:, :, None]
    quad = np.squeeze(err[:, None, :] @ sol, axis=(1, 2))
    dim = err.shape[1]
    return float(np.mean(quad) / max(dim, 1))


def test_block_nees_all_cells():
    n = 2000
    seed = 17
    out = {}
    for cell in CELLS:
        data = simulate_cell(seed, cell, n=n)
        mu, cov = fuse_cell(data, "sciu")
        err = mu - data["z"]
        nees_o = _block_nees(err[:, :1], cov[:, :1, :1])
        nees_p = _block_nees(err[:, 1:], cov[:, 1:, 1:])
        nees = _block_nees(err, cov)
        out[cell] = (nees_o, nees_p, nees)
        assert nees_o < 1.20, (cell, "overlap", nees_o)
        assert nees_p < 1.20, (cell, "complement", nees_p)
        assert nees < 1.20, (cell, "anees", nees)
    return out


def test_average_loewner_is_not_the_theorem():
    data = simulate_cell(17, "both", n=2000)
    mu, cov = fuse_cell(data, "sciu")
    err = mu - data["z"]
    mse = (err[:, :, None] * err[:, None, :]).mean(axis=0)
    covm = cov.mean(axis=0)
    gap = 0.5 * ((covm - mse) + (covm - mse).T)
    min_eig = float(np.min(np.linalg.eigvalsh(gap)))
    # May be slightly negative. That does not refute block NEES.
    return min_eig


if __name__ == "__main__":
    nees = test_block_nees_all_cells()
    joint_min = test_average_loewner_is_not_the_theorem()
    print("block_nees_ok", nees, "avg_loewner_min_eig", joint_min)

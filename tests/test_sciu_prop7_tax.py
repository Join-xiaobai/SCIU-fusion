"""Algebraic licence for Proposition 7: detector-free overlap cover is CU.

CPU only. Does not retune the 2x2 DGP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.srcf_filter import (  # noqa: E402
    _sym,
    cu_fuse_pair,
    det_optimal_ci_weight,
    sciu_fuse_pair,
    _pd_inverse,
)


def _ci_pair(mu_a, cov_a, mu_b, cov_b):
    ya = _pd_inverse(cov_a)
    yb = _pd_inverse(cov_b)
    omega = det_optimal_ci_weight(ya, yb)
    yfus = omega * ya + (1.0 - omega) * yb
    cov = _pd_inverse(yfus)
    mu = cov @ (omega * ya @ mu_a.reshape(-1, 1) + (1.0 - omega) * yb @ mu_b.reshape(-1, 1))
    return mu.reshape(-1), _sym(cov)


def _covers(U, mu_u, P, mu):
    gap = _sym(U - P - np.outer(mu_u - mu, mu_u - mu))
    return float(np.min(np.linalg.eigvalsh(gap)))


def test_sciu_overlap_is_cu():
    mu_a = np.array([0.0, 1.0, 0.2])
    mu_b = np.array([2.5, 0.1, -0.4])
    pd_a = 0.25 * np.eye(3)
    pd_b = 0.30 * np.eye(3)
    pi_a = np.diag([0.4, 0.5, 0.8])
    pi_b = np.diag([0.5, 0.7, 0.4])
    mu_u, p_u, _ = sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0,))
    mu_c, p_c, _ = cu_fuse_pair(
        mu_a[:1], (pd_a + pi_a)[:1, :1], mu_b[:1], (pd_b + pi_b)[:1, :1]
    )
    assert np.allclose(mu_u.reshape(-1)[:1], mu_c.reshape(-1), atol=1e-8)
    assert np.allclose(p_u[:1, :1], p_c, atol=1e-8)


def test_ci_fails_cu_constraint_on_conflict():
    mu_a = np.array([0.2])
    mu_b = np.array([-2.4])
    cov_a = np.array([[0.18]])
    cov_b = np.array([[0.22]])
    mu_ci, p_ci = _ci_pair(mu_a, cov_a, mu_b, cov_b)
    mu_cu, p_cu, _ = cu_fuse_pair(mu_a, cov_a, mu_b, cov_b)
    e_a_ci = _covers(p_ci, mu_ci, cov_a, mu_a.reshape(-1))
    e_b_ci = _covers(p_ci, mu_ci, cov_b, mu_b.reshape(-1))
    e_a_cu = _covers(p_cu, mu_cu.reshape(-1), cov_a, mu_a.reshape(-1))
    e_b_cu = _covers(p_cu, mu_cu.reshape(-1), cov_b, mu_b.reshape(-1))
    assert min(e_a_ci, e_b_ci) < -1e-6
    assert e_a_cu > -1e-8
    assert e_b_cu > -1e-8


if __name__ == "__main__":
    test_sciu_overlap_is_cu()
    test_ci_fails_cu_constraint_on_conflict()
    print("sciu_prop7_tax_ok")

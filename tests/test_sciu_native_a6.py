"""CPU identities for native-block SCIU (A6 revision).

The frozen restriction SCIU in test_sciu_identities.py is not withdrawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.srcf_filter import (  # noqa: E402
    batch_sciu_native_fuse_pair,
    native_block_joseph,
    sciu_native_fuse_pair,
    split_kalman_local,
)


H_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
H_B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_native_complement_ignores_overlap_y():
    mu = np.zeros(3)
    p = 1.2 * np.eye(3)
    y_a = np.array([9.0, 0.4])
    y_b = np.array([-7.0, -0.2])
    r_a = np.diag([0.25**2, 0.25**2])
    r_b = np.diag([0.35**2, 0.35**2])
    mu_ap, pd_ap, pi_ap = native_block_joseph(mu, p, y_a, H_A, r_a, np.array([1, 2]))
    mu_ap2, pd_ap2, pi_ap2 = native_block_joseph(
        mu, p, np.array([0.0, 0.4]), H_A, r_a, np.array([1, 2])
    )
    assert np.allclose(mu_ap, mu_ap2, atol=1e-12)
    assert np.allclose(pd_ap, pd_ap2, atol=1e-12)
    assert np.allclose(pi_ap, pi_ap2, atol=1e-12)
    mu_bp, _, _ = native_block_joseph(mu, p, y_b, H_B, r_b, np.array([1, 2]))
    mu_bp2, _, _ = native_block_joseph(
        mu, p, np.array([0.0, -0.2]), H_B, r_b, np.array([1, 2])
    )
    assert np.allclose(mu_bp, mu_bp2, atol=1e-12)


def test_native_overlap_ignores_private_y():
    mu = np.zeros(3)
    p = 1.2 * np.eye(3)
    r_a = np.diag([0.25**2, 0.25**2])
    mu_o, pd_o, pi_o = native_block_joseph(mu, p, np.array([0.3, 8.0]), H_A, r_a, np.array([0]))
    mu_o2, pd_o2, pi_o2 = native_block_joseph(
        mu, p, np.array([0.3, 0.0]), H_A, r_a, np.array([0])
    )
    assert np.allclose(mu_o, mu_o2, atol=1e-12)
    assert np.allclose(pd_o, pd_o2, atol=1e-12)
    assert np.allclose(pi_o, pi_o2, atol=1e-12)


def test_native_block_matches_restricted_h():
    mu = np.array([0.1, -0.2, 0.3])
    p = 1.2 * np.eye(3)
    y = np.array([0.5, -0.4])
    r = np.diag([0.25**2, 0.25**2])
    mu_n, pd_n, pi_n = native_block_joseph(mu, p, y, H_A, r, np.array([1, 2]))
    mu_r, pd_r, pi_r = split_kalman_local(
        mu[1:], p[1:, 1:], y[1:], np.array([[1.0, 0.0]]), np.array([[0.25**2]])
    )
    assert np.allclose(mu_n.reshape(-1), mu_r.reshape(-1), atol=1e-12)
    assert np.allclose(pd_n, pd_r, atol=1e-12)
    assert np.allclose(pi_n, pi_r, atol=1e-12)


def test_batch_native_matches_pair():
    rng = np.random.default_rng(3)
    n = 6
    mu = np.zeros((n, 3))
    p = np.repeat(1.2 * np.eye(3)[None, :, :], n, axis=0)
    y_a = rng.normal(size=(n, 2))
    y_b = rng.normal(size=(n, 2))
    r_a = np.repeat(np.diag([0.25**2, 0.25**2])[None, :, :], n, axis=0)
    r_b = np.repeat(np.diag([0.35**2, 0.35**2])[None, :, :], n, axis=0)
    mu_bch, p_bch, w_bch = batch_sciu_native_fuse_pair(
        mu, p, y_a, H_A, r_a, y_b, H_B, r_b, overlap_idx=(0,)
    )
    for i in range(n):
        mu_i, p_i, w_i = sciu_native_fuse_pair(
            mu[i], p[i], y_a[i], H_A, r_a[i], y_b[i], H_B, r_b[i], overlap_idx=(0,)
        )
        assert np.allclose(mu_i.reshape(-1), mu_bch[i], atol=1e-8)
        assert np.allclose(p_i, p_bch[i], atol=1e-8)
        assert abs(w_i - w_bch[i]) < 1e-12
        assert abs(p_bch[i, 0, 1]) < 1e-12
        assert abs(p_bch[i, 0, 2]) < 1e-12


if __name__ == "__main__":
    test_native_complement_ignores_overlap_y()
    test_native_overlap_ignores_private_y()
    test_native_block_matches_restricted_h()
    test_batch_native_matches_pair()
    print("sciu_native_a6_ok")

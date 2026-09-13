"""CPU identities for Split Covariance Intersection-Union (Path B).

No MIMIC, no GPU, no v6 DGP retune. Path A SRCF identities stay in
test_srcf_sci_identities.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.srcf_filter import (  # noqa: E402
    SequentialReliabilityConflictFilter,
    SourcePacket,
    batch_sciu_fuse_pair,
    cu_fuse_pair,
    sci_fuse_pair,
    sciu_fuse_pair,
    split_kalman_local,
    _sym,
)


def test_empty_overlap_is_sci():
    rng = np.random.default_rng(1)
    mu_a = rng.normal(size=3)
    mu_b = rng.normal(size=3)
    pd_a = _sym(rng.normal(size=(3, 3)))
    pd_a = pd_a @ pd_a.T + 0.2 * np.eye(3)
    pd_b = _sym(rng.normal(size=(3, 3)))
    pd_b = pd_b @ pd_b.T + 0.3 * np.eye(3)
    pi_a = np.diag([0.4, 0.5, 0.6])
    pi_b = np.diag([0.7, 0.4, 0.5])
    mu_s, p_s, w_s = sci_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b)
    mu_u, p_u, w_u = sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=())
    assert np.allclose(mu_s.reshape(-1), mu_u.reshape(-1), atol=1e-8)
    assert np.allclose(p_s, p_u, atol=1e-8)
    assert abs(w_s - w_u) < 1e-12


def test_empty_complement_is_cu():
    rng = np.random.default_rng(2)
    mu_a = rng.normal(size=2)
    mu_b = rng.normal(size=2)
    pd_a = 0.3 * np.eye(2)
    pd_b = 0.4 * np.eye(2)
    pi_a = np.diag([0.5, 0.8])
    pi_b = np.diag([0.9, 0.4])
    p_a = pd_a + pi_a
    p_b = pd_b + pi_b
    mu_c, p_c, w_c = cu_fuse_pair(mu_a, p_a, mu_b, p_b)
    mu_u, p_u, w_u = sciu_fuse_pair(
        mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0, 1)
    )
    assert np.allclose(mu_c.reshape(-1), mu_u.reshape(-1), atol=1e-8)
    assert np.allclose(p_c, p_u, atol=1e-8)
    assert abs(w_c - w_u) < 1e-12


def test_sciu_is_not_full_state_cu_with_private_coords():
    mu_a = np.array([0.0, 1.0, 0.0])
    mu_b = np.array([2.0, 0.0, -1.0])
    pd_a = 0.25 * np.eye(3)
    pd_b = 0.25 * np.eye(3)
    pi_a = np.diag([0.4, 0.5, 0.8])
    pi_b = np.diag([0.4, 0.8, 0.5])
    mu_u, p_u, _ = sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0,))
    mu_c, p_c, _ = cu_fuse_pair(mu_a, pd_a + pi_a, mu_b, pd_b + pi_b)
    assert abs(np.linalg.det(p_u)) < abs(np.linalg.det(p_c)) - 1e-8
    assert np.linalg.norm(p_u[0, 1:]) < 1e-12
    assert np.linalg.norm(p_u[1:, 0]) < 1e-12
    assert not np.allclose(mu_u.reshape(-1), mu_c.reshape(-1), atol=1e-4)


def test_batch_sciu_matches_pair():
    rng = np.random.default_rng(4)
    n = 5
    dim = 3
    mu_a = rng.normal(size=(n, dim))
    mu_b = rng.normal(size=(n, dim))
    pd_a = np.repeat(0.3 * np.eye(dim)[None, :, :], n, axis=0)
    pd_b = np.repeat(0.4 * np.eye(dim)[None, :, :], n, axis=0)
    pi_a = np.repeat(np.diag([0.5, 0.6, 0.7])[None, :, :], n, axis=0)
    pi_b = np.repeat(np.diag([0.7, 0.5, 0.6])[None, :, :], n, axis=0)
    mu_bch, p_bch, w_bch = batch_sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b)
    for i in range(n):
        mu, p, w = sciu_fuse_pair(mu_a[i], pd_a[i], pi_a[i], mu_b[i], pd_b[i], pi_b[i])
        assert np.allclose(mu.reshape(-1), mu_bch[i], atol=1e-8)
        assert np.allclose(p, p_bch[i], atol=1e-8)
        assert abs(w - w_bch[i]) < 1e-12


def test_sciu_filter_rule_runs():
    filt = SequentialReliabilityConflictFilter(3, fusion_rule="sciu", overlap_idx=(0,))
    a = SourcePacket(
        "A",
        y=np.array([0.2, -0.1]),
        H=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        available=True,
        base_var=0.3,
    )
    b = SourcePacket(
        "B",
        y=np.array([1.4, 0.9]),
        H=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        available=True,
        base_var=0.5,
    )
    state = filt.fuse_sequence([a, b], predict_first=True)
    assert state["cov"].shape == (3, 3)
    assert np.isfinite(state["mu"]).all()
    assert np.all(np.linalg.eigvalsh(state["cov"]) > 0)
    assert abs(state["cov"][0, 1]) < 1e-12
    assert abs(state["cov"][0, 2]) < 1e-12


def test_joseph_split_plus_sciu_corr_only_not_overconfident():
    """Corr-only sanity: common noise on overlap, no fault. SCIU ANEES ~<= CI."""
    rng = np.random.default_rng(8)
    n = 600
    dim = 3
    p_pred = 1.2 * np.eye(dim)
    mu_pred = np.zeros((n, dim))
    z = rng.multivariate_normal(np.zeros(dim), p_pred, size=n)
    common = rng.normal(0.0, 0.40, size=n)
    y_a = np.column_stack([z[:, 0] + common + rng.normal(0.0, 0.25, n), z[:, 1] + rng.normal(0.0, 0.25, n)])
    y_b = np.column_stack([z[:, 0] + common + rng.normal(0.0, 0.35, n), z[:, 2] + rng.normal(0.0, 0.35, n)])
    H_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    H_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r_a = np.diag([0.25 ** 2 + 0.40 ** 2, 0.25 ** 2])
    r_b = np.diag([0.35 ** 2 + 0.40 ** 2, 0.35 ** 2])
    p_pred_b = np.repeat(p_pred[None, :, :], n, axis=0)
    from src.models.srcf_filter import batch_split_kalman_local, batch_cu_fuse_pair

    mu_a, pd_a, pi_a = batch_split_kalman_local(mu_pred, p_pred_b, y_a, H_a, np.repeat(r_a[None], n, axis=0))
    mu_b, pd_b, pi_b = batch_split_kalman_local(mu_pred, p_pred_b, y_b, H_b, np.repeat(r_b[None], n, axis=0))
    mu_u, cov_u, _ = batch_sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0,))
    mu_c, cov_c, _ = batch_cu_fuse_pair(mu_a, pd_a + pi_a, mu_b, pd_b + pi_b)

    def _anees(err, cov):
        sol = np.linalg.solve(cov, err[:, :, None])
        quad = np.squeeze(err[:, None, :] @ sol, axis=(1, 2))
        return float(np.mean(quad) / err.shape[1])

    anees_u = _anees(mu_u - z, cov_u)
    logdet_u = float(np.mean(np.log(np.clip(np.abs(np.linalg.det(cov_u)), 1e-18, None))))
    logdet_c = float(np.mean(np.log(np.clip(np.abs(np.linalg.det(cov_c)), 1e-18, None))))
    assert anees_u < 1.20
    assert logdet_u < logdet_c - 1e-4


if __name__ == "__main__":
    test_empty_overlap_is_sci()
    test_empty_complement_is_cu()
    test_sciu_is_not_full_state_cu_with_private_coords()
    test_batch_sciu_matches_pair()
    test_sciu_filter_rule_runs()
    test_joseph_split_plus_sciu_corr_only_not_overconfident()
    print("sciu_identities_ok")

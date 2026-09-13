"""Sequential Reliability-Conflict Fusion operator.

Information Fusion object: a sequential covariance-intersection (CI) family
update for asynchronous sources with unknown cross-correlation.

Algebra, two sources against a common prediction:

    Y = Y_pred + omega Y_A + (1 - omega) Y_B

This is CI of the full-state estimates (Y_pred + Y_A) and (Y_pred + Y_B).
Independent Kalman instead adds both information matrices and is optimistic
when A and B share process / prediction error. Det-optimal CI chooses omega
in (0, 1) to minimise det(P). SRCF stays in the same simplex and only moves
omega toward the already-fused state when the incoming innovation is
conflicted.

Split CI (SCI) is a required comparator, not the paper object. After a Kalman
update of the common prediction, the local covariance splits into a dependent
part inherited from the prediction and an independent part from the
measurement (Li 2013; Cros et al. 2025). SCI treats only the dependent part
as unknown-correlation. Family-optimal OCI / SDP (paper 4718) is out of
scope: omega remains a 19-point grid. Inverse CI (Noack et al. 2017)
and Covariance Union (Bochardt et al. 2006) are required Information
Fusion comparators, not the paper object.

Split Covariance Intersection-Union (SCIU, Path B) is a different object.
It covers the product set unknown-correlation x exclusive-hypothesis by
unioning overlap marginals (CU) and split-intersecting the complement
(SCI). SCIU is not CI, not ICI, not SCI, not full-state CU, and not
SRCF. Sequential Reliability-Conflict Fusion remains the Path A fallback.

Sequential rule for more than two sources (standard sequential CI):

1. Predict, store (Y_pred, y_pred).
2. First available source is a Kalman update against the prediction
   (nested: det-optimal CI of Y_pred and Y_pred+Y_m collapses to Kalman).
3. Later sources are CI-combined with the current fused estimate:
   CI(current, Y_pred + Y_m).

Unavailable sources contribute zero information. Delay inflates source noise
before Y_m is formed. The older inflate-Kalman rule (Y += g Y_m) is kept as
`fusion_rule="inflate"` for historical ablation and is not the paper object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_col(x, dim):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"expected length {dim}, got {arr.size}")
    return arr.reshape(dim, 1)


def _pd_inverse(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _sym(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def _ci_grid(grid=None):
    if grid is None:
        return np.linspace(0.05, 0.95, 19)
    return np.asarray(grid, dtype=np.float64)


def det_optimal_ci_weight(lambda_a, lambda_b, grid=None):
    """Weight omega on estimate A that minimises det(P) for CI(A, B).

    Lambda(omega) = omega Lambda_A + (1 - omega) Lambda_B.
    Any omega in (0, 1) is a consistent CI combination under unknown
    cross-correlation. The grid only selects the least conservative member.
    """
    grid = _ci_grid(grid)
    lambda_a = _sym(lambda_a)
    lambda_b = _sym(lambda_b)
    best_omega = 0.5
    best_det = np.inf
    for omega in grid:
        fused = omega * lambda_a + (1.0 - omega) * lambda_b
        det_p = 1.0 / max(abs(np.linalg.det(fused)), 1e-18)
        if det_p < best_det:
            best_det = det_p
            best_omega = float(omega)
    return best_omega


def batch_det_optimal_ci_weight(lambda_a, lambda_b, grid=None):
    """Vectorised det-optimal CI weight on A. Arrays are (n, d, d)."""
    grid = _ci_grid(grid)
    lambda_a = np.asarray(lambda_a, dtype=np.float64)
    lambda_b = np.asarray(lambda_b, dtype=np.float64)
    dets = []
    for omega in grid:
        fused = omega * lambda_a + (1.0 - omega) * lambda_b
        det_lambda = np.linalg.det(fused)
        dets.append(1.0 / np.clip(np.abs(det_lambda), 1e-18, None))
    dets = np.stack(dets, axis=1)
    idx = np.argmin(dets, axis=1)
    return grid[idx]


def conflict_adaptive_ci_weight(omega_ci, conflict, df):
    """Increase the weight on the current fused estimate when conflicted.

    g = 1 / (1 + c / d) in (0, 1]. The pair remains a CI combination.
    """
    g = 1.0 / (1.0 + float(conflict) / max(float(df), 1.0))
    g = float(np.clip(g, 0.0, 1.0))
    omega_current = float(omega_ci) + (1.0 - float(omega_ci)) * (1.0 - g)
    return float(np.clip(omega_current, 0.0, 1.0)), g


def split_kalman_local(mu_pred, p_pred, y, H, R):
    """Joseph-form Kalman update of a common prediction, with SCI split.

    Dependent part Pd is inherited from the prediction. Independent part Pi
    comes from the measurement. P = Pd + Pi equals the Kalman covariance.
    """
    mu_pred = np.asarray(mu_pred, dtype=np.float64).reshape(-1, 1)
    p_pred = _sym(p_pred)
    H = np.asarray(H, dtype=np.float64)
    y = _as_col(y, H.shape[0])
    R = _sym(R)
    s = _sym(H @ p_pred @ H.T + R)
    try:
        k = p_pred @ H.T @ np.linalg.solve(s, np.eye(s.shape[0]))
    except np.linalg.LinAlgError:
        k = p_pred @ H.T @ _pd_inverse(s)
    i_kh = np.eye(p_pred.shape[0]) - k @ H
    mu = mu_pred + k @ (y - H @ mu_pred)
    pd = _sym(i_kh @ p_pred @ i_kh.T)
    pi = _sym(k @ R @ k.T)
    return mu, pd, pi


def sci_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, grid=None):
    """Det-optimal split covariance intersection of two split estimates.

    P_i(w) = Pd_i / w_i + Pi_i, then information-form Kalman of the inflated
    covariances. Not family-optimal OCI. Grid is the same 19-point CI grid.
    Returns fused mean, covariance, and weight on A.
    """
    grid = _ci_grid(grid)
    mu_a = np.asarray(mu_a, dtype=np.float64).reshape(-1, 1)
    mu_b = np.asarray(mu_b, dtype=np.float64).reshape(-1, 1)
    pd_a = _sym(pd_a)
    pi_a = _sym(pi_a)
    pd_b = _sym(pd_b)
    pi_b = _sym(pi_b)
    best_omega = 0.5
    best_det = np.inf
    best_mu = 0.5 * (mu_a + mu_b)
    best_p = _sym(pd_a + pi_a)
    for omega in grid:
        p1 = pd_a / max(float(omega), 1e-8) + pi_a
        p2 = pd_b / max(1.0 - float(omega), 1e-8) + pi_b
        y1 = _pd_inverse(p1)
        y2 = _pd_inverse(p2)
        fused_p = _pd_inverse(y1 + y2)
        det_p = abs(np.linalg.det(fused_p))
        if det_p < best_det:
            best_det = det_p
            best_omega = float(omega)
            best_p = _sym(fused_p)
            best_mu = best_p @ (y1 @ mu_a + y2 @ mu_b)
    return best_mu, best_p, best_omega


def batch_split_kalman_local(mu_pred, p_pred, y, H, R):
    """Batched Joseph-form local estimates. Shapes: (n, d), (n, d, d)."""
    n = mu_pred.shape[0]
    dim = mu_pred.shape[1]
    H = np.asarray(H, dtype=np.float64)
    obs = H.shape[0]
    y_col = np.asarray(y, dtype=np.float64).reshape(n, obs, 1)
    mu_col = np.asarray(mu_pred, dtype=np.float64).reshape(n, dim, 1)
    s = H[None, :, :] @ p_pred @ H.T[None, :, :] + R
    s = 0.5 * (s + np.transpose(s, (0, 2, 1)))
    try:
        k = p_pred @ H.T[None, :, :] @ np.linalg.inv(s)
    except np.linalg.LinAlgError:
        k = p_pred @ H.T[None, :, :] @ np.linalg.pinv(s)
    innov = y_col - H[None, :, :] @ mu_col
    mu = (mu_col + k @ innov).reshape(n, dim)
    i_kh = np.eye(dim)[None, :, :] - k @ H[None, :, :]
    pd = i_kh @ p_pred @ np.transpose(i_kh, (0, 2, 1))
    pi = k @ R @ np.transpose(k, (0, 2, 1))
    pd = 0.5 * (pd + np.transpose(pd, (0, 2, 1)))
    pi = 0.5 * (pi + np.transpose(pi, (0, 2, 1)))
    return mu, pd, pi


def batch_sci_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, grid=None):
    """Vectorised det-optimal SCI. Arrays are (n, d) and (n, d, d)."""
    grid = _ci_grid(grid)
    n, dim = mu_a.shape
    mu_a_col = mu_a.reshape(n, dim, 1)
    mu_b_col = mu_b.reshape(n, dim, 1)
    dets = []
    mus = []
    covs = []
    for omega in grid:
        p1 = pd_a / max(float(omega), 1e-8) + pi_a
        p2 = pd_b / max(1.0 - float(omega), 1e-8) + pi_b
        try:
            y1 = np.linalg.inv(p1)
            y2 = np.linalg.inv(p2)
            fused_p = np.linalg.inv(y1 + y2)
        except np.linalg.LinAlgError:
            y1 = np.linalg.pinv(p1)
            y2 = np.linalg.pinv(p2)
            fused_p = np.linalg.pinv(y1 + y2)
        fused_p = 0.5 * (fused_p + np.transpose(fused_p, (0, 2, 1)))
        mu = fused_p @ (y1 @ mu_a_col + y2 @ mu_b_col)
        dets.append(np.abs(np.linalg.det(fused_p)))
        mus.append(mu)
        covs.append(fused_p)
    dets = np.stack(dets, axis=1)
    idx = np.argmin(dets, axis=1)
    mus = np.stack(mus, axis=1)
    covs = np.stack(covs, axis=1)
    gather = np.arange(n)
    return (
        mus[gather, idx].reshape(n, dim),
        covs[gather, idx],
        grid[idx],
    )


def ici_fuse_pair(mu_a, cov_a, mu_b, cov_b, grid=None):
    """Det-optimal inverse covariance intersection (Noack et al., Automatica 2017, Thm 7).

    C^{-1} = C_A^{-1} + C_B^{-1} - (omega C_A + (1-omega) C_B)^{-1}.
    This is the unknown-common-information control, not sequential CI and
    not SRCF. Grid is the same 19-point CI grid.
    """
    grid = _ci_grid(grid)
    mu_a = np.asarray(mu_a, dtype=np.float64).reshape(-1, 1)
    mu_b = np.asarray(mu_b, dtype=np.float64).reshape(-1, 1)
    cov_a = _sym(cov_a)
    cov_b = _sym(cov_b)
    ya = _pd_inverse(cov_a)
    yb = _pd_inverse(cov_b)
    best_omega = 0.5
    best_det = np.inf
    best_mu = 0.5 * (mu_a + mu_b)
    best_p = _sym(0.5 * (cov_a + cov_b))
    for omega in grid:
        mix = _sym(float(omega) * cov_a + (1.0 - float(omega)) * cov_b)
        ymix = _pd_inverse(mix)
        yfus = _sym(ya + yb - ymix)
        fused_p = _sym(_pd_inverse(yfus))
        k = fused_p @ (ya - float(omega) * ymix)
        lmat = fused_p @ (yb - (1.0 - float(omega)) * ymix)
        fused_mu = k @ mu_a + lmat @ mu_b
        det_p = abs(np.linalg.det(fused_p))
        if det_p < best_det:
            best_det = det_p
            best_omega = float(omega)
            best_p = fused_p
            best_mu = fused_mu
    return best_mu, best_p, best_omega


def _min_det_psd_cover(f1, f2):
    """Min-det SPD matrix dominating two SPD matrices via simultaneous diag."""
    f1 = _sym(f1)
    f2 = _sym(f2)
    w, v = np.linalg.eigh(f1)
    w = np.clip(w, 1e-12, None)
    f1_sqrt = (v * np.sqrt(w)) @ v.T
    f1_inv_sqrt = (v * (1.0 / np.sqrt(w))) @ v.T
    mid = _sym(f1_inv_sqrt @ f2 @ f1_inv_sqrt)
    d, q = np.linalg.eigh(mid)
    scale = np.maximum(1.0, d)
    covered = f1_sqrt @ q @ np.diag(scale) @ q.T @ f1_sqrt
    return _sym(covered)


def cu_fuse_pair(mu_a, cov_a, mu_b, cov_b, grid=None):
    """Convex-combination Covariance Union (Bochardt et al., Fusion 2006).

    u = omega a + (1-omega) b, then the min-det U that satisfies
    U ≽ A+(u-a)(u-a)^T and U ≽ B+(u-b)(u-b)^T. Dual of CI under
    hypothesis conflict; not SRCF's in-simplex omega.
    """
    grid = _ci_grid(grid)
    mu_a = np.asarray(mu_a, dtype=np.float64).reshape(-1, 1)
    mu_b = np.asarray(mu_b, dtype=np.float64).reshape(-1, 1)
    cov_a = _sym(cov_a)
    cov_b = _sym(cov_b)
    delta = mu_b - mu_a
    best_omega = 0.5
    best_det = np.inf
    best_mu = 0.5 * (mu_a + mu_b)
    best_p = _sym(cov_a + cov_b + delta @ delta.T)
    for omega in grid:
        fused_mu = float(omega) * mu_a + (1.0 - float(omega)) * mu_b
        f1 = _sym(cov_a + ((1.0 - float(omega)) ** 2) * (delta @ delta.T))
        f2 = _sym(cov_b + (float(omega) ** 2) * (delta @ delta.T))
        fused_p = _min_det_psd_cover(f1, f2)
        det_p = abs(np.linalg.det(fused_p))
        if det_p < best_det:
            best_det = det_p
            best_omega = float(omega)
            best_p = fused_p
            best_mu = fused_mu
    return best_mu, best_p, best_omega


def batch_ici_fuse_pair(mu_a, cov_a, mu_b, cov_b, grid=None):
    """Vectorised det-optimal ICI. Arrays are (n, d) and (n, d, d)."""
    grid = _ci_grid(grid)
    n, dim = mu_a.shape
    mu_a_col = np.asarray(mu_a, dtype=np.float64).reshape(n, dim, 1)
    mu_b_col = np.asarray(mu_b, dtype=np.float64).reshape(n, dim, 1)
    cov_a = np.asarray(cov_a, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)
    try:
        ya = np.linalg.inv(cov_a)
        yb = np.linalg.inv(cov_b)
    except np.linalg.LinAlgError:
        ya = np.linalg.pinv(cov_a)
        yb = np.linalg.pinv(cov_b)
    dets = []
    mus = []
    covs = []
    for omega in grid:
        mix = omega * cov_a + (1.0 - omega) * cov_b
        try:
            ymix = np.linalg.inv(mix)
            yfus = ya + yb - ymix
            fused_p = np.linalg.inv(yfus)
        except np.linalg.LinAlgError:
            ymix = np.linalg.pinv(mix)
            yfus = ya + yb - ymix
            fused_p = np.linalg.pinv(yfus)
        fused_p = 0.5 * (fused_p + np.transpose(fused_p, (0, 2, 1)))
        k = fused_p @ (ya - omega * ymix)
        lmat = fused_p @ (yb - (1.0 - omega) * ymix)
        mu = k @ mu_a_col + lmat @ mu_b_col
        dets.append(np.abs(np.linalg.det(fused_p)))
        mus.append(mu)
        covs.append(fused_p)
    dets = np.stack(dets, axis=1)
    idx = np.argmin(dets, axis=1)
    mus = np.stack(mus, axis=1)
    covs = np.stack(covs, axis=1)
    gather = np.arange(n)
    return mus[gather, idx].reshape(n, dim), covs[gather, idx], grid[idx]


def _batch_min_det_psd_cover(f1, f2):
    f1 = 0.5 * (f1 + np.transpose(f1, (0, 2, 1)))
    f2 = 0.5 * (f2 + np.transpose(f2, (0, 2, 1)))
    w, v = np.linalg.eigh(f1)
    w = np.clip(w, 1e-12, None)
    f1_sqrt = np.matmul(v * np.sqrt(w)[:, None, :], np.transpose(v, (0, 2, 1)))
    f1_inv_sqrt = np.matmul(v * (1.0 / np.sqrt(w))[:, None, :], np.transpose(v, (0, 2, 1)))
    mid = np.matmul(np.matmul(f1_inv_sqrt, f2), f1_inv_sqrt)
    mid = 0.5 * (mid + np.transpose(mid, (0, 2, 1)))
    d, q = np.linalg.eigh(mid)
    scale = np.maximum(1.0, d)
    inner = np.matmul(q * scale[:, None, :], np.transpose(q, (0, 2, 1)))
    covered = np.matmul(np.matmul(f1_sqrt, inner), np.transpose(f1_sqrt, (0, 2, 1)))
    return 0.5 * (covered + np.transpose(covered, (0, 2, 1)))


def batch_cu_fuse_pair(mu_a, cov_a, mu_b, cov_b, grid=None):
    """Vectorised convex-combination CU. Arrays are (n, d) and (n, d, d)."""
    grid = _ci_grid(grid)
    n, dim = mu_a.shape
    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64)
    cov_a = np.asarray(cov_a, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)
    delta = mu_b - mu_a
    outer = delta[:, :, None] * delta[:, None, :]
    dets = []
    mus = []
    covs = []
    for omega in grid:
        fused_mu = omega * mu_a + (1.0 - omega) * mu_b
        f1 = cov_a + ((1.0 - omega) ** 2) * outer
        f2 = cov_b + (omega ** 2) * outer
        fused_p = _batch_min_det_psd_cover(f1, f2)
        dets.append(np.abs(np.linalg.det(fused_p)))
        mus.append(fused_mu)
        covs.append(fused_p)
    dets = np.stack(dets, axis=1)
    idx = np.argmin(dets, axis=1)
    mus = np.stack(mus, axis=1)
    covs = np.stack(covs, axis=1)
    gather = np.arange(n)
    return mus[gather, idx], covs[gather, idx], grid[idx]


def _complement_idx(dim, overlap_idx):
    overlap = set(int(i) for i in np.asarray(overlap_idx, dtype=int).reshape(-1))
    return np.array([i for i in range(int(dim)) if i not in overlap], dtype=int)


def sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0,), grid=None):
    """Split Covariance Intersection-Union of two Joseph locals (Path B).

    Overlap coordinates are Covariance-Unioned from the full local
    marginals. Complement coordinates are Split-CI fused from the Joseph
    split. Off-block cross-covariance is zero. Identities:

    - empty overlap -> SCI of the two locals;
    - empty complement -> CU of the two full-state locals;
    - not full-state CU when the complement is non-empty.

    Returns fused mean, covariance, and the overlap CU weight on A.
    """
    mu_a = np.asarray(mu_a, dtype=np.float64).reshape(-1)
    mu_b = np.asarray(mu_b, dtype=np.float64).reshape(-1)
    if mu_a.size != mu_b.size:
        raise ValueError("SCIU locals must share state dimension")
    dim = mu_a.size
    overlap_idx = np.asarray(overlap_idx, dtype=int).reshape(-1)
    complement_idx = _complement_idx(dim, overlap_idx)
    pd_a = _sym(pd_a)
    pi_a = _sym(pi_a)
    pd_b = _sym(pd_b)
    pi_b = _sym(pi_b)
    p_a = _sym(pd_a + pi_a)
    p_b = _sym(pd_b + pi_b)
    if overlap_idx.size == 0:
        return sci_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, grid=grid)
    if complement_idx.size == 0:
        return cu_fuse_pair(mu_a, p_a, mu_b, p_b, grid=grid)

    mu_ao = mu_a[overlap_idx]
    mu_bo = mu_b[overlap_idx]
    pa_o = p_a[np.ix_(overlap_idx, overlap_idx)]
    pb_o = p_b[np.ix_(overlap_idx, overlap_idx)]
    u_o, U_o, omega_o = cu_fuse_pair(mu_ao, pa_o, mu_bo, pb_o, grid=grid)

    mu_ap = mu_a[complement_idx]
    mu_bp = mu_b[complement_idx]
    pd_ap = pd_a[np.ix_(complement_idx, complement_idx)]
    pi_ap = pi_a[np.ix_(complement_idx, complement_idx)]
    pd_bp = pd_b[np.ix_(complement_idx, complement_idx)]
    pi_bp = pi_b[np.ix_(complement_idx, complement_idx)]
    u_p, P_p, _omega_p = sci_fuse_pair(mu_ap, pd_ap, pi_ap, mu_bp, pd_bp, pi_bp, grid=grid)

    fused_mu = np.zeros((dim, 1), dtype=np.float64)
    fused_p = np.zeros((dim, dim), dtype=np.float64)
    fused_mu[overlap_idx, 0] = np.asarray(u_o, dtype=np.float64).reshape(-1)
    fused_mu[complement_idx, 0] = np.asarray(u_p, dtype=np.float64).reshape(-1)
    fused_p[np.ix_(overlap_idx, overlap_idx)] = _sym(U_o)
    fused_p[np.ix_(complement_idx, complement_idx)] = _sym(P_p)
    return fused_mu, _sym(fused_p), float(omega_o)


def batch_sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0,), grid=None):
    """Vectorised SCIU. Arrays are (n, d) and (n, d, d)."""
    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64)
    pd_a = np.asarray(pd_a, dtype=np.float64)
    pi_a = np.asarray(pi_a, dtype=np.float64)
    pd_b = np.asarray(pd_b, dtype=np.float64)
    pi_b = np.asarray(pi_b, dtype=np.float64)
    n, dim = mu_a.shape
    overlap_idx = np.asarray(overlap_idx, dtype=int).reshape(-1)
    complement_idx = _complement_idx(dim, overlap_idx)
    p_a = 0.5 * ((pd_a + pi_a) + np.transpose(pd_a + pi_a, (0, 2, 1)))
    p_b = 0.5 * ((pd_b + pi_b) + np.transpose(pd_b + pi_b, (0, 2, 1)))
    if overlap_idx.size == 0:
        return batch_sci_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, grid=grid)
    if complement_idx.size == 0:
        return batch_cu_fuse_pair(mu_a, p_a, mu_b, p_b, grid=grid)

    mu_o, cov_o, omega_o = batch_cu_fuse_pair(
        mu_a[:, overlap_idx],
        p_a[:, overlap_idx[:, None], overlap_idx],
        mu_b[:, overlap_idx],
        p_b[:, overlap_idx[:, None], overlap_idx],
        grid=grid,
    )
    mu_p, cov_p, _omega_p = batch_sci_fuse_pair(
        mu_a[:, complement_idx],
        pd_a[:, complement_idx[:, None], complement_idx],
        pi_a[:, complement_idx[:, None], complement_idx],
        mu_b[:, complement_idx],
        pd_b[:, complement_idx[:, None], complement_idx],
        pi_b[:, complement_idx[:, None], complement_idx],
        grid=grid,
    )
    fused_mu = np.zeros((n, dim), dtype=np.float64)
    fused_p = np.zeros((n, dim, dim), dtype=np.float64)
    fused_mu[:, overlap_idx] = mu_o
    fused_mu[:, complement_idx] = mu_p
    fused_p[:, overlap_idx[:, None], overlap_idx] = cov_o
    fused_p[:, complement_idx[:, None], complement_idx] = cov_p
    fused_p = 0.5 * (fused_p + np.transpose(fused_p, (0, 2, 1)))
    return fused_mu, fused_p, omega_o


def _obs_rows_in_block(H, block_idx):
    keep = set(int(i) for i in np.asarray(block_idx, dtype=int).reshape(-1))
    H = np.asarray(H, dtype=np.float64)
    rows = []
    for r in range(H.shape[0]):
        support = set(np.flatnonzero(np.abs(H[r]) > 1e-12).tolist())
        if support and support.issubset(keep):
            rows.append(r)
    return np.asarray(rows, dtype=int)


def _slice_r(R, rows, n=None):
    rows = np.asarray(rows, dtype=int)
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        sliced = R[np.ix_(rows, rows)]
        if n is None:
            return _sym(sliced)
        return np.repeat(sliced[None, :, :], n, axis=0)
    return R[:, rows[:, None], rows]


def native_block_joseph(mu_pred, p_pred, y, H, R, block_idx):
    """Joseph update on a coordinate block using only measurements of that block."""
    block_idx = np.asarray(block_idx, dtype=int).reshape(-1)
    if block_idx.size == 0:
        raise ValueError("empty block")
    mu_pred = np.asarray(mu_pred, dtype=np.float64).reshape(-1)
    p_pred = _sym(p_pred)
    rows = _obs_rows_in_block(H, block_idx)
    mu_b = mu_pred[block_idx]
    p_b = p_pred[np.ix_(block_idx, block_idx)]
    if rows.size == 0:
        return mu_b, p_b, np.zeros_like(p_b)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    H = np.asarray(H, dtype=np.float64)
    y_b = y[rows]
    H_b = H[np.ix_(rows, block_idx)]
    R_b = _slice_r(R, rows)
    return split_kalman_local(mu_b, p_b, y_b, H_b, R_b)


def batch_native_block_joseph(mu_pred, p_pred, y, H, R, block_idx):
    """Batched native Joseph on one coordinate block. y is (n, obs)."""
    block_idx = np.asarray(block_idx, dtype=int).reshape(-1)
    n = mu_pred.shape[0]
    if block_idx.size == 0:
        raise ValueError("empty block")
    rows = _obs_rows_in_block(H, block_idx)
    mu_b = mu_pred[:, block_idx]
    p_b = p_pred[:, block_idx[:, None], block_idx]
    if rows.size == 0:
        return mu_b, p_b, np.zeros_like(p_b)
    y_b = np.asarray(y, dtype=np.float64)[:, rows]
    H_b = np.asarray(H, dtype=np.float64)[np.ix_(rows, block_idx)]
    R_b = _slice_r(R, rows, n=n)
    return batch_split_kalman_local(mu_b, p_b, y_b, H_b, R_b)


def sciu_native_fuse_pair(
    mu_pred,
    p_pred,
    y_a,
    H_a,
    R_a,
    y_b,
    H_b,
    R_b,
    overlap_idx=(0,),
    grid=None,
):
    """SCIU with native Joseph splits on overlap and complement.

    Overlap measurements update only S_o. Private measurements update
    only S_perp. Then CU on overlap locals and SCI on complement locals.
    This is the A6 revision: the complement gain never sees overlap y.
    Empty overlap recovers SCI of native full-state Joseph locals.
    Empty complement recovers CU of native overlap Joseph locals.
    """
    mu_pred = np.asarray(mu_pred, dtype=np.float64).reshape(-1)
    p_pred = _sym(p_pred)
    dim = mu_pred.size
    overlap_idx = np.asarray(overlap_idx, dtype=int).reshape(-1)
    complement_idx = _complement_idx(dim, overlap_idx)

    def _full_native():
        mu_a, pd_a, pi_a = split_kalman_local(mu_pred, p_pred, y_a, H_a, R_a)
        mu_b, pd_b, pi_b = split_kalman_local(mu_pred, p_pred, y_b, H_b, R_b)
        return sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=overlap_idx, grid=grid)

    if overlap_idx.size == 0 or complement_idx.size == 0:
        return _full_native()

    mu_ao, pd_ao, pi_ao = native_block_joseph(mu_pred, p_pred, y_a, H_a, R_a, overlap_idx)
    mu_bo, pd_bo, pi_bo = native_block_joseph(mu_pred, p_pred, y_b, H_b, R_b, overlap_idx)
    u_o, U_o, omega_o = cu_fuse_pair(mu_ao, _sym(pd_ao + pi_ao), mu_bo, _sym(pd_bo + pi_bo), grid=grid)

    mu_ap, pd_ap, pi_ap = native_block_joseph(mu_pred, p_pred, y_a, H_a, R_a, complement_idx)
    mu_bp, pd_bp, pi_bp = native_block_joseph(mu_pred, p_pred, y_b, H_b, R_b, complement_idx)
    u_p, P_p, _ = sci_fuse_pair(mu_ap, pd_ap, pi_ap, mu_bp, pd_bp, pi_bp, grid=grid)

    fused_mu = np.zeros((dim, 1), dtype=np.float64)
    fused_p = np.zeros((dim, dim), dtype=np.float64)
    fused_mu[overlap_idx, 0] = np.asarray(u_o, dtype=np.float64).reshape(-1)
    fused_mu[complement_idx, 0] = np.asarray(u_p, dtype=np.float64).reshape(-1)
    fused_p[np.ix_(overlap_idx, overlap_idx)] = _sym(U_o)
    fused_p[np.ix_(complement_idx, complement_idx)] = _sym(P_p)
    return fused_mu, _sym(fused_p), float(omega_o)


def batch_sciu_native_fuse_pair(
    mu_pred,
    p_pred,
    y_a,
    H_a,
    R_a,
    y_b,
    H_b,
    R_b,
    overlap_idx=(0,),
    grid=None,
):
    """Vectorised native-block SCIU. y_* are (n, obs); R_* are (n, obs, obs) or (obs, obs)."""
    mu_pred = np.asarray(mu_pred, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    n, dim = mu_pred.shape
    overlap_idx = np.asarray(overlap_idx, dtype=int).reshape(-1)
    complement_idx = _complement_idx(dim, overlap_idx)
    if overlap_idx.size == 0 or complement_idx.size == 0:
        mu_a, pd_a, pi_a = batch_split_kalman_local(mu_pred, p_pred, y_a, H_a, R_a)
        mu_b, pd_b, pi_b = batch_split_kalman_local(mu_pred, p_pred, y_b, H_b, R_b)
        return batch_sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=overlap_idx, grid=grid)

    mu_ao, pd_ao, pi_ao = batch_native_block_joseph(mu_pred, p_pred, y_a, H_a, R_a, overlap_idx)
    mu_bo, pd_bo, pi_bo = batch_native_block_joseph(mu_pred, p_pred, y_b, H_b, R_b, overlap_idx)
    mu_o, cov_o, omega_o = batch_cu_fuse_pair(mu_ao, pd_ao + pi_ao, mu_bo, pd_bo + pi_bo, grid=grid)
    mu_ap, pd_ap, pi_ap = batch_native_block_joseph(mu_pred, p_pred, y_a, H_a, R_a, complement_idx)
    mu_bp, pd_bp, pi_bp = batch_native_block_joseph(mu_pred, p_pred, y_b, H_b, R_b, complement_idx)
    mu_p, cov_p, _ = batch_sci_fuse_pair(mu_ap, pd_ap, pi_ap, mu_bp, pd_bp, pi_bp, grid=grid)

    fused_mu = np.zeros((n, dim), dtype=np.float64)
    fused_p = np.zeros((n, dim, dim), dtype=np.float64)
    fused_mu[:, overlap_idx] = mu_o
    fused_mu[:, complement_idx] = mu_p
    fused_p[:, overlap_idx[:, None], overlap_idx] = cov_o
    fused_p[:, complement_idx[:, None], complement_idx] = cov_p
    fused_p = 0.5 * (fused_p + np.transpose(fused_p, (0, 2, 1)))
    return fused_mu, fused_p, omega_o


@dataclass
class SourcePacket:
    name: str
    y: np.ndarray
    H: np.ndarray
    available: bool
    delay: float = 0.0
    base_var: float = 1.0
    delay_slope: float = 0.35


class SequentialReliabilityConflictFilter:
    """Sequential CI-family fusion with delay-dependent source reliability."""

    def __init__(
        self,
        state_dim,
        F=None,
        Q=None,
        prior_var=4.0,
        conflict_df_floor=1.0,
        conflict_enabled=True,
        delay_enabled=True,
        fusion_rule="srcf",
        overlap_idx=(0,),
    ):
        self.state_dim = int(state_dim)
        self.F = np.eye(self.state_dim, dtype=np.float64) if F is None else np.asarray(F, dtype=np.float64)
        self.Q = 0.25 * np.eye(self.state_dim, dtype=np.float64) if Q is None else np.asarray(Q, dtype=np.float64)
        self.prior_var = float(prior_var)
        self.conflict_df_floor = float(conflict_df_floor)
        self.conflict_enabled = bool(conflict_enabled)
        self.delay_enabled = bool(delay_enabled)
        allowed = {"kalman", "ci", "srcf", "inflate", "sci", "ici", "cu", "sciu"}
        if fusion_rule not in allowed:
            raise ValueError(f"fusion_rule must be one of {sorted(allowed)}")
        self.fusion_rule = fusion_rule
        self.overlap_idx = tuple(int(i) for i in np.asarray(overlap_idx, dtype=int).reshape(-1))
        self.reset()

    def reset(self, mu0=None, var0=None):
        dim = self.state_dim
        if mu0 is None:
            mu0 = np.zeros((dim, 1), dtype=np.float64)
        else:
            mu0 = _as_col(mu0, dim)
        var = self.prior_var if var0 is None else float(var0)
        cov = var * np.eye(dim, dtype=np.float64)
        self.Lambda = _pd_inverse(cov)
        self.eta = self.Lambda @ mu0
        self.lambda_pred = self.Lambda.copy()
        self.eta_pred = self.eta.copy()
        self.last_conflict = {}
        self.last_omega = {}
        self.last_g = {}
        return self.state()

    def state(self):
        cov = _pd_inverse(self.Lambda)
        mu = cov @ self.eta
        return {
            "mu": mu.reshape(-1),
            "cov": cov,
            "Lambda": self.Lambda.copy(),
            "uncertainty": float(np.trace(cov)),
            "conflict": dict(self.last_conflict),
            "omega": dict(self.last_omega),
            "agreement": dict(self.last_g),
            "fusion_rule": self.fusion_rule,
        }

    def predict(self):
        cov = _pd_inverse(self.Lambda)
        mu = cov @ self.eta
        pred_cov = _sym(self.F @ cov @ self.F.T + self.Q)
        self.Lambda = _pd_inverse(pred_cov)
        self.eta = self.Lambda @ (self.F @ mu)
        self.lambda_pred = self.Lambda.copy()
        self.eta_pred = self.eta.copy()
        return self.state()

    def _source_information(self, packet: SourcePacket):
        noise = float(packet.base_var)
        if self.delay_enabled:
            noise = noise * (1.0 + float(packet.delay_slope) * max(float(packet.delay), 0.0))
        H = np.asarray(packet.H, dtype=np.float64)
        y = _as_col(packet.y, H.shape[0])
        precision = (1.0 / max(noise, 1e-8)) * np.eye(H.shape[0], dtype=np.float64)
        lambda_m = _sym(H.T @ precision @ H)
        eta_m = H.T @ precision @ y
        return lambda_m, eta_m, y, H, precision, _pd_inverse(precision)

    def _innovation_conflict(self, y, H, precision):
        cov = _pd_inverse(self.Lambda)
        mu = cov @ self.eta
        nu = y - H @ mu
        S = _sym(H @ cov @ H.T + _pd_inverse(precision))
        try:
            conflict = float(np.squeeze(nu.T @ np.linalg.solve(S, nu)))
        except np.linalg.LinAlgError:
            conflict = float(np.squeeze(nu.T @ (_pd_inverse(S) @ nu)))
        return conflict

    def _sci_fuse_packets(self, packets):
        mu_pred = _pd_inverse(self.lambda_pred) @ self.eta_pred
        p_pred = _pd_inverse(self.lambda_pred)
        locals_ = []
        for packet in packets:
            name = packet.name
            if not packet.available:
                self.last_conflict[name] = np.nan
                self.last_omega[name] = 0.0
                self.last_g[name] = np.nan
                continue
            _, _, y, H, precision, r_mat = self._source_information(packet)
            conflict = self._innovation_conflict(y, H, precision)
            mu_l, pd_l, pi_l = split_kalman_local(mu_pred, p_pred, y, H, r_mat)
            locals_.append((name, mu_l, pd_l, pi_l, conflict, H.shape[0]))
            self.last_conflict[name] = conflict
            self.last_g[name] = 1.0
        if not locals_:
            return self.state()
        if len(locals_) == 1:
            name, mu_l, pd_l, pi_l, _, _ = locals_[0]
            cov = _sym(pd_l + pi_l)
            self.Lambda = _pd_inverse(cov)
            self.eta = self.Lambda @ mu_l
            self.last_omega[name] = 0.0
            return self.state()
        name_a, fused_mu, fused_pd, fused_pi, _, _ = locals_[0]
        omega_last = 0.5
        g = 1.0
        for name_b, mu_b, pd_b, pi_b, conflict_b, df_b in locals_[1:]:
            mu_f, p_f, omega = sci_fuse_pair(fused_mu, fused_pd, fused_pi, mu_b, pd_b, pi_b)
            if self.conflict_enabled and self.fusion_rule != "sci":
                omega, g = conflict_adaptive_ci_weight(omega, conflict_b, df_b)
                mu_f, p_f, _ = sci_fuse_pair(
                    fused_mu, fused_pd, fused_pi, mu_b, pd_b, pi_b, grid=np.array([omega])
                )
            else:
                g = 1.0
            fused_mu = mu_f
            fused_pd = _sym(p_f)
            fused_pi = np.zeros_like(fused_pd)
            omega_last = omega
            self.last_omega[name_b] = omega
            self.last_g[name_b] = g
        self.last_omega[name_a] = 1.0 - omega_last
        self.Lambda = _pd_inverse(_sym(fused_pd + fused_pi))
        self.eta = self.Lambda @ fused_mu
        return self.state()

    def _locals_from_prediction(self, packets):
        mu_pred = _pd_inverse(self.lambda_pred) @ self.eta_pred
        p_pred = _pd_inverse(self.lambda_pred)
        locals_ = []
        for packet in packets:
            name = packet.name
            if not packet.available:
                self.last_conflict[name] = np.nan
                self.last_omega[name] = 0.0
                self.last_g[name] = np.nan
                continue
            _, _, y, H, precision, r_mat = self._source_information(packet)
            conflict = self._innovation_conflict(y, H, precision)
            mu_l, pd_l, pi_l = split_kalman_local(mu_pred, p_pred, y, H, r_mat)
            cov_l = _sym(pd_l + pi_l)
            locals_.append((name, mu_l, cov_l, conflict))
            self.last_conflict[name] = conflict
            self.last_g[name] = 1.0
        return locals_

    def _ici_fuse_packets(self, packets):
        locals_ = self._locals_from_prediction(packets)
        if not locals_:
            return self.state()
        if len(locals_) == 1:
            name, mu_l, cov_l, _ = locals_[0]
            self.Lambda = _pd_inverse(cov_l)
            self.eta = self.Lambda @ mu_l
            self.last_omega[name] = 0.0
            return self.state()
        name_a, fused_mu, fused_cov, _ = locals_[0]
        omega_last = 0.5
        for name_b, mu_b, cov_b, _conflict_b in locals_[1:]:
            mu_f, p_f, omega = ici_fuse_pair(fused_mu, fused_cov, mu_b, cov_b)
            fused_mu = mu_f
            fused_cov = _sym(p_f)
            omega_last = omega
            self.last_omega[name_b] = omega
            self.last_g[name_b] = 1.0
        self.last_omega[name_a] = 1.0 - omega_last
        self.Lambda = _pd_inverse(_sym(fused_cov))
        self.eta = self.Lambda @ fused_mu
        return self.state()

    def _cu_fuse_packets(self, packets):
        locals_ = self._locals_from_prediction(packets)
        if not locals_:
            return self.state()
        if len(locals_) == 1:
            name, mu_l, cov_l, _ = locals_[0]
            self.Lambda = _pd_inverse(cov_l)
            self.eta = self.Lambda @ mu_l
            self.last_omega[name] = 0.0
            return self.state()
        name_a, fused_mu, fused_cov, _ = locals_[0]
        omega_last = 0.5
        for name_b, mu_b, cov_b, _conflict_b in locals_[1:]:
            mu_f, p_f, omega = cu_fuse_pair(fused_mu, fused_cov, mu_b, cov_b)
            fused_mu = mu_f
            fused_cov = _sym(p_f)
            omega_last = omega
            self.last_omega[name_b] = omega
            self.last_g[name_b] = 1.0
        self.last_omega[name_a] = 1.0 - omega_last
        self.Lambda = _pd_inverse(_sym(fused_cov))
        self.eta = self.Lambda @ fused_mu
        return self.state()

    def _sciu_fuse_packets(self, packets):
        """Path B: overlap CU + complement SCI of Joseph locals."""
        mu_pred = _pd_inverse(self.lambda_pred) @ self.eta_pred
        p_pred = _pd_inverse(self.lambda_pred)
        locals_ = []
        for packet in packets:
            name = packet.name
            if not packet.available:
                self.last_conflict[name] = np.nan
                self.last_omega[name] = 0.0
                self.last_g[name] = np.nan
                continue
            _, _, y, H, precision, r_mat = self._source_information(packet)
            conflict = self._innovation_conflict(y, H, precision)
            mu_l, pd_l, pi_l = split_kalman_local(mu_pred, p_pred, y, H, r_mat)
            locals_.append((name, mu_l, pd_l, pi_l, conflict))
            self.last_conflict[name] = conflict
            self.last_g[name] = 1.0
        if not locals_:
            return self.state()
        if len(locals_) == 1:
            name, mu_l, pd_l, pi_l, _ = locals_[0]
            cov = _sym(pd_l + pi_l)
            self.Lambda = _pd_inverse(cov)
            self.eta = self.Lambda @ mu_l
            self.last_omega[name] = 0.0
            return self.state()
        name_a, fused_mu, fused_pd, fused_pi, _ = locals_[0]
        omega_last = 0.5
        for name_b, mu_b, pd_b, pi_b, _conflict_b in locals_[1:]:
            mu_f, p_f, omega = sciu_fuse_pair(
                fused_mu,
                fused_pd,
                fused_pi,
                mu_b,
                pd_b,
                pi_b,
                overlap_idx=self.overlap_idx,
            )
            fused_mu = mu_f
            fused_pd = _sym(p_f)
            fused_pi = np.zeros_like(fused_pd)
            omega_last = omega
            self.last_omega[name_b] = omega
            self.last_g[name_b] = 1.0
        self.last_omega[name_a] = 1.0 - omega_last
        self.Lambda = _pd_inverse(_sym(fused_pd + fused_pi))
        self.eta = self.Lambda @ fused_mu
        return self.state()

    def update(self, packet: SourcePacket):

        name = packet.name
        if not packet.available:
            self.last_conflict[name] = np.nan
            self.last_omega[name] = 0.0
            self.last_g[name] = np.nan
            return self.state()
        lambda_m, eta_m, y, H, precision, _ = self._source_information(packet)
        conflict = self._innovation_conflict(y, H, precision)
        df = max(float(H.shape[0]), self.conflict_df_floor)

        if self.fusion_rule == "kalman":
            omega = 0.0
            g = 1.0
            self.Lambda = _sym(self.Lambda + lambda_m)
            self.eta = self.eta + eta_m
        elif self.fusion_rule == "inflate":
            g = 1.0 / (1.0 + conflict / df) if self.conflict_enabled else 1.0
            omega = float(np.clip(g, 0.0, 1.0))
            self.Lambda = _sym(self.Lambda + omega * lambda_m)
            self.eta = self.eta + omega * eta_m
        else:
            lambda_local = _sym(self.lambda_pred + lambda_m)
            eta_local = self.eta_pred + eta_m
            omega_ci = det_optimal_ci_weight(self.Lambda, lambda_local)
            if self.fusion_rule == "ci" or not self.conflict_enabled:
                omega = omega_ci
                g = 1.0
            else:
                omega, g = conflict_adaptive_ci_weight(omega_ci, conflict, df)
            self.Lambda = _sym(omega * self.Lambda + (1.0 - omega) * lambda_local)
            self.eta = omega * self.eta + (1.0 - omega) * eta_local

        self.last_conflict[name] = conflict
        self.last_omega[name] = omega
        self.last_g[name] = g
        return self.state()

    def fuse_sequence(self, packets, predict_first=True):
        if predict_first:
            self.predict()
        if self.fusion_rule == "sci":
            return self._sci_fuse_packets(packets)
        if self.fusion_rule == "ici":
            return self._ici_fuse_packets(packets)
        if self.fusion_rule == "cu":
            return self._cu_fuse_packets(packets)
        if self.fusion_rule == "sciu":
            return self._sciu_fuse_packets(packets)
        for packet in packets:
            self.update(packet)
        return self.state()


def risk_from_state(mu, uncertainty, weights, bias=0.0):
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    feat = np.concatenate([mu, [np.log1p(max(float(uncertainty), 0.0))]])
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size != feat.size:
        raise ValueError("risk weights must match [mu; log(1+u)]")
    logit = float(weights @ feat + bias)
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))

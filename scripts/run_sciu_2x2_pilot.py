#!/usr/bin/env python3
"""Aggregate-only 2x2 known-truth gate for Split Covariance Intersection-Union.

Path B object. Path A SRCF and the v6 clinical DGP are not this generator.
No simulated rows. No GPU. No concat-Brier hunt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.srcf_filter import (  # noqa: E402
    batch_cu_fuse_pair,
    batch_det_optimal_ci_weight,
    batch_ici_fuse_pair,
    batch_sci_fuse_pair,
    batch_sciu_fuse_pair,
    batch_split_kalman_local,
    conflict_adaptive_ci_weight,
)


CELLS = ("none", "corr_only", "conflict_only", "both")
OPERATORS = ("kalman", "ci", "sci", "ici", "cu", "sciu", "srcf")
H_A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
H_B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
SIGMA_A = 0.25
SIGMA_B = 0.35
COMMON_SD = 0.40
FLIP_SD = 0.15
P_PRED = 1.2


def coverage_95(err, cov_diag):
    half = 1.959963984540054 * np.sqrt(np.clip(cov_diag, 1e-8, None))
    return float(np.mean(np.abs(err) <= half))


def anees_score(err, cov):
    err = np.asarray(err, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    try:
        sol = np.linalg.solve(cov, err[:, :, None])
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(cov) @ err[:, :, None]
    quad = np.squeeze(err[:, None, :] @ sol, axis=(1, 2))
    dim = err.shape[1]
    return float(np.mean(quad) / max(dim, 1))


def simulate_cell(seed, cell, n=4000, dim=3):
    if cell not in CELLS:
        raise ValueError(cell)
    rng = np.random.default_rng(seed + 1000 * (CELLS.index(cell) + 1))
    p_pred = P_PRED * np.eye(dim)
    z = rng.multivariate_normal(np.zeros(dim), p_pred, size=n)
    y_a = z @ H_A.T + rng.normal(0.0, SIGMA_A, size=(n, 2))
    y_b = z @ H_B.T + rng.normal(0.0, SIGMA_B, size=(n, 2))
    use_corr = cell in {"corr_only", "both"}
    use_conflict = cell in {"conflict_only", "both"}
    if use_corr:
        common = rng.normal(0.0, COMMON_SD, size=n)
        y_a[:, 0] = y_a[:, 0] + common
        y_b[:, 0] = y_b[:, 0] + common
    conflict = np.zeros(n, dtype=bool)
    if use_conflict:
        conflict = np.ones(n, dtype=bool)
        y_b = y_b.copy()
        y_b[:, 0] = -y_b[:, 0] + rng.normal(0.0, FLIP_SD, size=n)
    return {
        "z": z,
        "y_a": y_a,
        "y_b": y_b,
        "conflict": conflict,
        "p_pred": p_pred,
        "cell": cell,
        "use_corr": use_corr,
        "use_conflict": use_conflict,
    }


def _pd_inv(mats):
    try:
        return np.linalg.inv(mats)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mats)


def _cell_r(use_corr):
    extra = COMMON_SD ** 2 if use_corr else 0.0
    r_a = np.diag([SIGMA_A ** 2 + extra, SIGMA_A ** 2])
    r_b = np.diag([SIGMA_B ** 2 + extra, SIGMA_B ** 2])
    return r_a, r_b


def fuse_cell(data, fusion_rule):
    n = data["z"].shape[0]
    dim = data["z"].shape[1]
    p_pred = np.repeat(data["p_pred"][None, :, :], n, axis=0)
    mu_pred = np.zeros((n, dim), dtype=np.float64)
    r_a, r_b = _cell_r(data["use_corr"])
    r_a_b = np.repeat(r_a[None, :, :], n, axis=0)
    r_b_b = np.repeat(r_b[None, :, :], n, axis=0)
    mu_a, pd_a, pi_a = batch_split_kalman_local(mu_pred, p_pred, data["y_a"], H_A, r_a_b)
    mu_b, pd_b, pi_b = batch_split_kalman_local(mu_pred, p_pred, data["y_b"], H_B, r_b_b)
    cov_a = pd_a + pi_a
    cov_b = pd_b + pi_b

    if fusion_rule == "kalman":
        ya = data["y_a"].reshape(n, 2, 1)
        yb = data["y_b"].reshape(n, 2, 1)
        inf_pred = _pd_inv(p_pred)
        inf_a = H_A.T @ np.linalg.inv(r_a) @ H_A
        inf_b = H_B.T @ np.linalg.inv(r_b) @ H_B
        lambda_k = inf_pred + inf_a[None, :, :] + inf_b[None, :, :]
        cov = _pd_inv(lambda_k)
        eta = inf_pred @ mu_pred.reshape(n, dim, 1)
        eta = eta + (H_A.T @ np.linalg.inv(r_a) @ ya) + (H_B.T @ np.linalg.inv(r_b) @ yb)
        mu = (cov @ eta).reshape(n, dim)
        return mu, cov
    if fusion_rule == "ci":
        ya = _pd_inv(cov_a)
        yb = _pd_inv(cov_b)
        omega = batch_det_optimal_ci_weight(ya, yb)
        w = omega[:, None, None]
        yfus = w * ya + (1.0 - w) * yb
        cov = _pd_inv(yfus)
        mu_a_c = mu_a.reshape(n, dim, 1)
        mu_b_c = mu_b.reshape(n, dim, 1)
        mu = (cov @ (w * ya @ mu_a_c + (1.0 - w) * yb @ mu_b_c)).reshape(n, dim)
        return mu, cov
    if fusion_rule == "sci":
        mu, cov, _ = batch_sci_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b)
        return mu, cov
    if fusion_rule == "ici":
        mu, cov, _ = batch_ici_fuse_pair(mu_a, cov_a, mu_b, cov_b)
        return mu, cov
    if fusion_rule == "cu":
        mu, cov, _ = batch_cu_fuse_pair(mu_a, cov_a, mu_b, cov_b)
        return mu, cov
    if fusion_rule == "sciu":
        mu, cov, _ = batch_sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0,))
        return mu, cov
    if fusion_rule == "srcf":
        ya = _pd_inv(cov_a)
        yb = _pd_inv(cov_b)
        omega_ci = batch_det_optimal_ci_weight(ya, yb)
        nu = (data["y_b"][:, 0] - mu_a[:, 0]).reshape(-1)
        s = cov_a[:, 0, 0] + r_b[0, 0]
        conflict = (nu ** 2) / np.clip(s, 1e-8, None)
        omega = np.empty(n, dtype=np.float64)
        for i in range(n):
            omega[i], _g = conflict_adaptive_ci_weight(float(omega_ci[i]), float(conflict[i]), 1.0)
        w = omega[:, None, None]
        yfus = w * ya + (1.0 - w) * yb
        cov = _pd_inv(yfus)
        mu_a_c = mu_a.reshape(n, dim, 1)
        mu_b_c = mu_b.reshape(n, dim, 1)
        mu = (cov @ (w * ya @ mu_a_c + (1.0 - w) * yb @ mu_b_c)).reshape(n, dim)
        return mu, cov
    raise ValueError(fusion_rule)


def metrics(z, mu, cov):
    err = mu - z
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    cov_diag = np.diagonal(cov, axis1=1, axis2=2)
    logdet = np.log(np.clip(np.abs(np.linalg.det(cov)), 1e-18, None))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "overlap_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "private_rmse": float(np.sqrt(np.mean(err[:, 1:] ** 2))),
        "coverage95": coverage_95(err, cov_diag),
        "anees": anees_score(err, cov),
        "overlap_anees": anees_score(err[:, :1], cov[:, :1, :1]),
        "mean_logdet": float(np.mean(logdet)),
        "n": int(len(z)),
    }


def evaluate_seed(seed, n, operators):
    out = {"seed": int(seed), "cells": {}}
    for cell in CELLS:
        data = simulate_cell(seed, cell, n=n)
        cell_out = {}
        for rule in operators:
            mu, cov = fuse_cell(data, rule)
            cell_out[rule] = metrics(data["z"], mu, cov)
        out["cells"][cell] = cell_out
    return out


def gate_from_means(means):
    sciu_anees = {cell: means[cell]["sciu"]["anees"] for cell in CELLS}
    # Catalogue failure is on the overlap coordinate. Full-state ANEES is
    # diluted by private Kalman-correct coordinates and is not the test.
    kalman_corr_overlap = means["corr_only"]["kalman"]["overlap_anees"]
    ci_conf = means["conflict_only"]["ci"]["anees"]
    sci_conf = means["conflict_only"]["sci"]["anees"]
    logdet_ok = (
        means["corr_only"]["sciu"]["mean_logdet"] < means["corr_only"]["cu"]["mean_logdet"]
        and means["both"]["sciu"]["mean_logdet"] < means["both"]["cu"]["mean_logdet"]
    )
    private_ok = (
        means["corr_only"]["sciu"]["private_rmse"] < means["corr_only"]["cu"]["private_rmse"]
        and means["both"]["sciu"]["private_rmse"] < means["both"]["cu"]["private_rmse"]
    )
    pass_anees = all(v <= 1.15 for v in sciu_anees.values())
    pass_kalman = kalman_corr_overlap > 1.20
    pass_conflict_catalogue = (ci_conf > 1.20) or (sci_conf > 1.20)
    passed = bool(pass_anees and pass_kalman and pass_conflict_catalogue and logdet_ok and private_ok)
    return {
        "sciu_anees_le_1_15_every_cell": pass_anees,
        "kalman_overlap_anees_gt_1_20_corr_only": pass_kalman,
        "ci_or_sci_anees_gt_1_20_conflict_only": pass_conflict_catalogue,
        "sciu_logdet_lt_cu_corr_and_both": logdet_ok,
        "sciu_private_rmse_lt_cu_corr_and_both": private_ok,
        "passed": passed,
        "sciu_anees": sciu_anees,
        "kalman_corr_only_overlap_anees": kalman_corr_overlap,
        "ci_conflict_only_anees": ci_conf,
        "sci_conflict_only_anees": sci_conf,
    }


def mean_over_seeds(seed_rows):
    means = {}
    for cell in CELLS:
        means[cell] = {}
        for rule in OPERATORS:
            keys = seed_rows[0]["cells"][cell][rule].keys()
            means[cell][rule] = {}
            for key in keys:
                if key == "n":
                    means[cell][rule][key] = int(seed_rows[0]["cells"][cell][rule][key])
                    continue
                vals = [row["cells"][cell][rule][key] for row in seed_rows]
                means[cell][rule][key] = float(np.mean(vals))
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "results" / "2x2"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = [evaluate_seed(seed, args.n, OPERATORS) for seed in args.seeds]
    means = mean_over_seeds(seed_rows)
    gate = gate_from_means(means)
    payload = {
        "object": "sciu",
        "path": "B",
        "fallback": "srcf_path_a",
        "n_per_cell": int(args.n),
        "seeds": [int(s) for s in args.seeds],
        "cells": list(CELLS),
        "operators": list(OPERATORS),
        "gpu_authorized": False,
        "seed_rows": seed_rows,
        "means": means,
        "gate": gate,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# SCIU 2x2 known-truth gate",
        "",
        f"n={args.n} per cell; seeds={list(args.seeds)}; CPU only.",
        f"passed={gate['passed']}",
        "",
        "| cell | kalman ANEES | CI | SCI | ICI | CU | SCIU | SRCF | SCIU logdet | CU logdet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        row = means[cell]
        lines.append(
            "| {cell} | {kalman:.3f} | {ci:.3f} | {sci:.3f} | {ici:.3f} | {cu:.3f} | {sciu:.3f} | {srcf:.3f} | {ldu:.3f} | {ldc:.3f} |".format(
                cell=cell,
                kalman=row["kalman"]["anees"],
                ci=row["ci"]["anees"],
                sci=row["sci"]["anees"],
                ici=row["ici"]["anees"],
                cu=row["cu"]["anees"],
                sciu=row["sciu"]["anees"],
                srcf=row["srcf"]["anees"],
                ldu=row["sciu"]["mean_logdet"],
                ldc=row["cu"]["mean_logdet"],
            )
        )
    lines.extend(
        [
            "",
            f"- SCIU ANEES <= 1.15 every cell: {gate['sciu_anees_le_1_15_every_cell']}",
            f"- Kalman overlap ANEES > 1.20 corr-only: {gate['kalman_overlap_anees_gt_1_20_corr_only']}",
            f"- CI or SCI ANEES > 1.20 conflict-only: {gate['ci_or_sci_anees_gt_1_20_conflict_only']}",
            f"- SCIU logdet < CU on corr-only and both: {gate['sciu_logdet_lt_cu_corr_and_both']}",
            f"- SCIU private RMSE < CU on corr-only and both: {gate['sciu_private_rmse_lt_cu_corr_and_both']}",
            "",
            "Path A SRCF is recorded, not the Path B object. v6 DGP was not retuned.",
        ]
    )
    (out_dir / "GATE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": gate["passed"], "sciu_anees": gate["sciu_anees"]}, indent=2))


if __name__ == "__main__":
    main()

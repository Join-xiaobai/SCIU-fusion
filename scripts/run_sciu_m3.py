#!/usr/bin/env python3
"""Sequential three-source support for SCIU. CPU only.

Not a retune of the frozen two-source 2x2 gate.
Conflict is a sign-flip of source B overlap only. Order A then B then C.
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
    batch_sci_fuse_pair,
    batch_sciu_fuse_pair,
    batch_split_kalman_local,
)

CELLS = ("none", "corr_only", "conflict_only", "both")
H_A = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
H_B = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
H_C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
SIGMA_A = 0.25
SIGMA_B = 0.35
SIGMA_C = 0.30
COMMON_SD = 0.40
FLIP_SD = 0.15
P_PRED = 1.2


def _pd_inv(mats):
    try:
        return np.linalg.inv(mats)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mats)


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
    return float(np.mean(quad) / max(err.shape[1], 1))


def cell_r(use_corr):
    extra = COMMON_SD ** 2 if use_corr else 0.0
    r_a = np.diag([SIGMA_A ** 2 + extra, SIGMA_A ** 2])
    r_b = np.diag([SIGMA_B ** 2 + extra, SIGMA_B ** 2])
    r_c = np.diag([SIGMA_C ** 2 + extra, SIGMA_C ** 2])
    return r_a, r_b, r_c


def simulate_cell(seed, cell, n=2000, dim=4):
    rng = np.random.default_rng(seed + 3000 * (CELLS.index(cell) + 1))
    p_pred = P_PRED * np.eye(dim)
    z = rng.multivariate_normal(np.zeros(dim), p_pred, size=n)
    y_a = z @ H_A.T + rng.normal(0.0, SIGMA_A, size=(n, 2))
    y_b = z @ H_B.T + rng.normal(0.0, SIGMA_B, size=(n, 2))
    y_c = z @ H_C.T + rng.normal(0.0, SIGMA_C, size=(n, 2))
    use_corr = cell in {"corr_only", "both"}
    use_conflict = cell in {"conflict_only", "both"}
    if use_corr:
        common = rng.normal(0.0, COMMON_SD, size=n)
        y_a[:, 0] += common
        y_b[:, 0] += common
        y_c[:, 0] += common
    if use_conflict:
        y_b = y_b.copy()
        y_b[:, 0] = -y_b[:, 0] + rng.normal(0.0, FLIP_SD, size=n)
    return {
        "z": z,
        "y_a": y_a,
        "y_b": y_b,
        "y_c": y_c,
        "p_pred": p_pred,
        "use_corr": use_corr,
        "use_conflict": use_conflict,
    }


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


def locals_abc(data):
    n = data["z"].shape[0]
    dim = data["z"].shape[1]
    p_pred = np.repeat(data["p_pred"][None, :, :], n, axis=0)
    mu_pred = np.zeros((n, dim))
    r_a, r_b, r_c = cell_r(data["use_corr"])
    mu_a, pd_a, pi_a = batch_split_kalman_local(
        mu_pred, p_pred, data["y_a"], H_A, np.repeat(r_a[None], n, axis=0)
    )
    mu_b, pd_b, pi_b = batch_split_kalman_local(
        mu_pred, p_pred, data["y_b"], H_B, np.repeat(r_b[None], n, axis=0)
    )
    mu_c, pd_c, pi_c = batch_split_kalman_local(
        mu_pred, p_pred, data["y_c"], H_C, np.repeat(r_c[None], n, axis=0)
    )
    return (mu_a, pd_a, pi_a), (mu_b, pd_b, pi_b), (mu_c, pd_c, pi_c)


def sequential_sciu(loc_a, loc_b, loc_c):
    mu_ab, p_ab, _ = batch_sciu_fuse_pair(*loc_a, *loc_b, overlap_idx=(0,))
    zeros = np.zeros_like(p_ab)
    mu, p, _ = batch_sciu_fuse_pair(mu_ab, p_ab, zeros, *loc_c, overlap_idx=(0,))
    return mu, p


def sequential_sci(loc_a, loc_b, loc_c):
    mu_ab, p_ab, _ = batch_sci_fuse_pair(*loc_a, *loc_b)
    zeros = np.zeros_like(p_ab)
    mu, p, _ = batch_sci_fuse_pair(mu_ab, p_ab, zeros, *loc_c)
    return mu, p


def sequential_cu(loc_a, loc_b, loc_c):
    cov_a = loc_a[1] + loc_a[2]
    cov_b = loc_b[1] + loc_b[2]
    cov_c = loc_c[1] + loc_c[2]
    mu_ab, p_ab, _ = batch_cu_fuse_pair(loc_a[0], cov_a, loc_b[0], cov_b)
    mu, p, _ = batch_cu_fuse_pair(mu_ab, p_ab, loc_c[0], cov_c)
    return mu, p


def sequential_ci(loc_a, loc_b, loc_c):
    cov_a = loc_a[1] + loc_a[2]
    cov_b = loc_b[1] + loc_b[2]
    cov_c = loc_c[1] + loc_c[2]
    ya = _pd_inv(cov_a)
    yb = _pd_inv(cov_b)
    omega = batch_det_optimal_ci_weight(ya, yb)
    w = omega[:, None, None]
    yfus = w * ya + (1.0 - w) * yb
    cov_ab = _pd_inv(yfus)
    mu_a = loc_a[0].reshape(len(loc_a[0]), 4, 1)
    mu_b = loc_b[0].reshape(len(loc_b[0]), 4, 1)
    mu_ab = (cov_ab @ (w * ya @ mu_a + (1.0 - w) * yb @ mu_b)).reshape(len(loc_a[0]), 4)
    yc = _pd_inv(cov_c)
    yab = _pd_inv(cov_ab)
    omega2 = batch_det_optimal_ci_weight(yab, yc)
    w2 = omega2[:, None, None]
    yfus2 = w2 * yab + (1.0 - w2) * yc
    cov = _pd_inv(yfus2)
    mu_c = loc_c[0].reshape(len(loc_c[0]), 4, 1)
    mu_ab_c = mu_ab.reshape(len(mu_ab), 4, 1)
    mu = (cov @ (w2 * yab @ mu_ab_c + (1.0 - w2) * yc @ mu_c)).reshape(len(mu_ab), 4)
    return mu, cov


def mean_over(seed_rows, operators):
    means = {}
    for cell in CELLS:
        means[cell] = {}
        for op in operators:
            keys = seed_rows[0]["cells"][cell][op].keys()
            means[cell][op] = {}
            for key in keys:
                if key == "n":
                    means[cell][op][key] = int(seed_rows[0]["cells"][cell][op][key])
                    continue
                means[cell][op][key] = float(
                    np.mean([row["cells"][cell][op][key] for row in seed_rows])
                )
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "results" / "m3"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    operators = ("ci", "sci", "cu", "sciu")
    seed_rows = []
    for seed in args.seeds:
        cells = {}
        for cell in CELLS:
            data = simulate_cell(seed, cell, n=args.n)
            loc_a, loc_b, loc_c = locals_abc(data)
            fused = {
                "ci": sequential_ci(loc_a, loc_b, loc_c),
                "sci": sequential_sci(loc_a, loc_b, loc_c),
                "cu": sequential_cu(loc_a, loc_b, loc_c),
                "sciu": sequential_sciu(loc_a, loc_b, loc_c),
            }
            cells[cell] = {
                name: metrics(data["z"], mu, cov) for name, (mu, cov) in fused.items()
            }
        seed_rows.append({"seed": int(seed), "cells": cells})
    means = mean_over(seed_rows, operators)
    sciu_anees = {cell: means[cell]["sciu"]["anees"] for cell in CELLS}
    gate = {
        "sciu_anees_le_1_15": all(v <= 1.15 for v in sciu_anees.values()),
        "ci_or_sci_conflict_gt_1_20": (
            means["conflict_only"]["ci"]["anees"] > 1.20
            or means["conflict_only"]["sci"]["anees"] > 1.20
        ),
        "sciu_private_lt_cu": (
            means["corr_only"]["sciu"]["private_rmse"]
            < means["corr_only"]["cu"]["private_rmse"]
            and means["both"]["sciu"]["private_rmse"]
            < means["both"]["cu"]["private_rmse"]
        ),
        "sciu_anees": sciu_anees,
        "ci_conflict": means["conflict_only"]["ci"]["anees"],
        "sci_conflict": means["conflict_only"]["sci"]["anees"],
    }
    gate["passed"] = bool(
        gate["sciu_anees_le_1_15"]
        and gate["ci_or_sci_conflict_gt_1_20"]
        and gate["sciu_private_lt_cu"]
    )
    payload = {
        "object": "sciu_m3",
        "path": "B",
        "gpu_authorized": False,
        "n_per_cell": int(args.n),
        "seeds": [int(s) for s in args.seeds],
        "order": "A then B then C",
        "conflict": "sign-flip of B overlap only",
        "seed_rows": seed_rows,
        "means": means,
        "gate": gate,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Sequential three-source SCIU support",
        "",
        f"n={args.n}; seeds={list(args.seeds)}; order A-B-C; CPU only.",
        f"passed={gate['passed']}",
        "",
        "| cell | CI ANEES | SCI | CU | SCIU | SCIU priv | CU priv |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        row = means[cell]
        lines.append(
            "| {cell} | {ci:.3f} | {sci:.3f} | {cu:.3f} | {sciu:.3f} | {sp:.3f} | {cp:.3f} |".format(
                cell=cell,
                ci=row["ci"]["anees"],
                sci=row["sci"]["anees"],
                cu=row["cu"]["anees"],
                sciu=row["sciu"]["anees"],
                sp=row["sciu"]["private_rmse"],
                cp=row["cu"]["private_rmse"],
            )
        )
    lines.extend(
        [
            "",
            f"- SCIU ANEES <= 1.15: {gate['sciu_anees_le_1_15']}",
            f"- CI or SCI conflict > 1.20: {gate['ci_or_sci_conflict_gt_1_20']}",
            f"- SCIU private RMSE < CU: {gate['sciu_private_lt_cu']}",
            "",
            "Not order-invariant. Frozen two-source 2x2 was not overwritten.",
        ]
    )
    (out_dir / "GATE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": gate["passed"], "sciu_anees": sciu_anees}, indent=2))


if __name__ == "__main__":
    main()

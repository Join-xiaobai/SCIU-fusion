#!/usr/bin/env python3
"""CPU A6 native-complement revision and I_o sensitivity.

Frozen 2x2 DGP from run_sciu_2x2_pilot.py is not retuned.
Original gate 20260912_sciu_2x2 is not overwritten.
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

from scripts.run_sciu_2x2_pilot import (  # noqa: E402
    CELLS,
    H_A,
    H_B,
    SIGMA_A,
    SIGMA_B,
    _cell_r,
    fuse_cell,
    metrics,
    simulate_cell,
)
from src.models.srcf_filter import batch_sciu_fuse_pair, batch_sciu_native_fuse_pair, batch_split_kalman_local  # noqa: E402


IO_CASES = {
    "true": (0,),
    "wrong": (1,),
    "extra": (0, 1),
    "empty": (),
}


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


def fuse_restriction(data, overlap_idx=(0,)):
    n = data["z"].shape[0]
    dim = data["z"].shape[1]
    p_pred = np.repeat(data["p_pred"][None, :, :], n, axis=0)
    mu_pred = np.zeros((n, dim), dtype=np.float64)
    r_a, r_b = _cell_r(data["use_corr"])
    r_a_b = np.repeat(r_a[None, :, :], n, axis=0)
    r_b_b = np.repeat(r_b[None, :, :], n, axis=0)
    mu_a, pd_a, pi_a = batch_split_kalman_local(mu_pred, p_pred, data["y_a"], H_A, r_a_b)
    mu_b, pd_b, pi_b = batch_split_kalman_local(mu_pred, p_pred, data["y_b"], H_B, r_b_b)
    mu, cov, _ = batch_sciu_fuse_pair(
        mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=overlap_idx
    )
    return mu, cov


def fuse_native(data, overlap_idx=(0,), p_pred=None):
    n = data["z"].shape[0]
    dim = data["z"].shape[1]
    if p_pred is None:
        p_pred = np.repeat(data["p_pred"][None, :, :], n, axis=0)
    mu_pred = np.zeros((n, dim), dtype=np.float64)
    r_a, r_b = _cell_r(data["use_corr"])
    r_a_b = np.repeat(r_a[None, :, :], n, axis=0)
    r_b_b = np.repeat(r_b[None, :, :], n, axis=0)
    mu, cov, _ = batch_sciu_native_fuse_pair(
        mu_pred,
        p_pred,
        data["y_a"],
        H_A,
        r_a_b,
        data["y_b"],
        H_B,
        r_b_b,
        overlap_idx=overlap_idx,
    )
    return mu, cov


def extra_metrics(z, mu, cov):
    out = metrics(z, mu, cov)
    err = mu - z
    out["complement_anees"] = _block_nees(err[:, 1:], cov[:, 1:, 1:])
    return out


def mean_dict(rows):
    keys = rows[0].keys()
    out = {}
    for key in keys:
        if key == "n":
            out[key] = int(rows[0][key])
            continue
        out[key] = float(np.mean([row[key] for row in rows]))
    return out


def evaluate_a6(seeds, n):
    seed_rows = []
    for seed in seeds:
        cells = {}
        for cell in CELLS:
            data = simulate_cell(seed, cell, n=n)
            mu_r, cov_r = fuse_restriction(data, overlap_idx=(0,))
            mu_n, cov_n = fuse_native(data, overlap_idx=(0,))
            mu_c, cov_c = fuse_cell(data, "cu")
            cells[cell] = {
                "sciu_restriction": extra_metrics(data["z"], mu_r, cov_r),
                "sciu_native": extra_metrics(data["z"], mu_n, cov_n),
                "cu": extra_metrics(data["z"], mu_c, cov_c),
            }
        seed_rows.append({"seed": int(seed), "cells": cells})
    means = {}
    for cell in CELLS:
        means[cell] = {
            name: mean_dict([row["cells"][cell][name] for row in seed_rows])
            for name in ("sciu_restriction", "sciu_native", "cu")
        }
    native_anees = {cell: means[cell]["sciu_native"]["anees"] for cell in CELLS}
    native_comp = {cell: means[cell]["sciu_native"]["complement_anees"] for cell in CELLS}
    logdet_ok = (
        means["corr_only"]["sciu_native"]["mean_logdet"]
        < means["corr_only"]["cu"]["mean_logdet"]
        and means["both"]["sciu_native"]["mean_logdet"]
        < means["both"]["cu"]["mean_logdet"]
    )
    private_ok = (
        means["corr_only"]["sciu_native"]["private_rmse"]
        < means["corr_only"]["cu"]["private_rmse"]
        and means["both"]["sciu_native"]["private_rmse"]
        < means["both"]["cu"]["private_rmse"]
    )
    gate = {
        "native_anees_le_1_15": all(v <= 1.15 for v in native_anees.values()),
        "native_complement_anees_le_1_15": all(v <= 1.15 for v in native_comp.values()),
        "native_logdet_lt_cu": logdet_ok,
        "native_private_rmse_lt_cu": private_ok,
        "native_anees": native_anees,
        "native_complement_anees": native_comp,
    }
    gate["passed"] = bool(
        gate["native_anees_le_1_15"]
        and gate["native_complement_anees_le_1_15"]
        and gate["native_logdet_lt_cu"]
        and gate["native_private_rmse_lt_cu"]
    )
    return seed_rows, means, gate


def evaluate_io(seeds, n):
    seed_rows = []
    for seed in seeds:
        cells = {}
        for cell in CELLS:
            data = simulate_cell(seed, cell, n=n)
            cells[cell] = {}
            for name, idx in IO_CASES.items():
                mu, cov = fuse_restriction(data, overlap_idx=idx)
                cells[cell][name] = extra_metrics(data["z"], mu, cov)
        seed_rows.append({"seed": int(seed), "cells": cells})
    means = {}
    for cell in CELLS:
        means[cell] = {
            name: mean_dict([row["cells"][cell][name] for row in seed_rows])
            for name in IO_CASES
        }
    true_ok = all(means[cell]["true"]["anees"] <= 1.15 for cell in CELLS)
    empty_conflict = means["conflict_only"]["empty"]["anees"]
    wrong_conflict = means["conflict_only"]["wrong"]["anees"]
    extra_priv = means["corr_only"]["extra"]["private_rmse"]
    true_priv = means["corr_only"]["true"]["private_rmse"]
    gate = {
        "true_anees_le_1_15": true_ok,
        "empty_conflict_anees": empty_conflict,
        "empty_fails_conflict": empty_conflict > 1.20,
        "wrong_conflict_anees": wrong_conflict,
        "wrong_fails_conflict": wrong_conflict > 1.20,
        "extra_private_rmse_corr_only": extra_priv,
        "true_private_rmse_corr_only": true_priv,
        "extra_pays_private_tax": extra_priv > true_priv,
    }
    gate["passed"] = bool(
        gate["true_anees_le_1_15"]
        and gate["empty_fails_conflict"]
        and gate["wrong_fails_conflict"]
        and gate["extra_pays_private_tax"]
    )
    return seed_rows, means, gate


def evaluate_coupled(seeds, n, rho=0.45):
    """Stress remainder A6: coupled prediction, not a 2x2 retune.

    Latent z is redrawn from the coupled P so the probe is not a
    misspecified filter on diagonal truth. Measurement noise law is
    the frozen 2x2 law.
    """
    seed_rows = []
    p = 1.2 * np.eye(3)
    p[0, 1] = p[1, 0] = rho
    p[0, 2] = p[2, 0] = rho
    from scripts.run_sciu_2x2_pilot import COMMON_SD, FLIP_SD
    for seed in seeds:
        cells = {}
        for cell_i, cell in enumerate(CELLS):
            rng = np.random.default_rng(int(seed) + 7000 + 1000 * (cell_i + 1))
            z = rng.multivariate_normal(np.zeros(3), p, size=n)
            y_a = z @ H_A.T + rng.normal(0.0, SIGMA_A, size=(n, 2))
            y_b = z @ H_B.T + rng.normal(0.0, SIGMA_B, size=(n, 2))
            use_corr = cell in {"corr_only", "both"}
            use_conflict = cell in {"conflict_only", "both"}
            if use_corr:
                common = rng.normal(0.0, COMMON_SD, size=n)
                y_a[:, 0] = y_a[:, 0] + common
                y_b[:, 0] = y_b[:, 0] + common
            if use_conflict:
                y_b = y_b.copy()
                y_b[:, 0] = -y_b[:, 0] + rng.normal(0.0, FLIP_SD, size=n)
            data = {
                "z": z,
                "y_a": y_a,
                "y_b": y_b,
                "p_pred": p,
                "use_corr": use_corr,
                "use_conflict": use_conflict,
            }
            n_obs = n
            p_pred = np.repeat(p[None, :, :], n_obs, axis=0)
            mu_pred = np.zeros((n_obs, 3))
            r_a, r_b = _cell_r(use_corr)
            r_a_b = np.repeat(r_a[None, :, :], n_obs, axis=0)
            r_b_b = np.repeat(r_b[None, :, :], n_obs, axis=0)
            mu_a, pd_a, pi_a = batch_split_kalman_local(mu_pred, p_pred, y_a, H_A, r_a_b)
            mu_b, pd_b, pi_b = batch_split_kalman_local(mu_pred, p_pred, y_b, H_B, r_b_b)
            mu_r, cov_r, _ = batch_sciu_fuse_pair(mu_a, pd_a, pi_a, mu_b, pd_b, pi_b, overlap_idx=(0,))
            mu_n, cov_n = fuse_native(data, overlap_idx=(0,), p_pred=p_pred)
            cells[cell] = {
                "restriction": extra_metrics(z, mu_r, cov_r),
                "native": extra_metrics(z, mu_n, cov_n),
            }
        seed_rows.append({"seed": int(seed), "cells": cells})
    means = {}
    for cell in CELLS:
        means[cell] = {
            name: mean_dict([row["cells"][cell][name] for row in seed_rows])
            for name in ("restriction", "native")
        }
    return seed_rows, means


def write_report(payload, out_dir):
    a6 = payload["a6"]["gate"]
    io = payload["io"]["gate"]
    lines = [
        "# SCIU A6 native complement and I_o sensitivity",
        "",
        f"n={payload['n_per_cell']}; seeds={payload['seeds']}; CPU only.",
        f"a6_passed={a6['passed']}",
        f"io_passed={io['passed']}",
        "",
        "## A. Native complement Joseph (A6 revision)",
        "",
        "| cell | restriction ANEES | native ANEES | native complement NEES | native priv RMSE | CU priv RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        row = payload["a6"]["means"][cell]
        lines.append(
            "| {cell} | {r:.3f} | {n:.3f} | {c:.3f} | {np:.3f} | {cp:.3f} |".format(
                cell=cell,
                r=row["sciu_restriction"]["anees"],
                n=row["sciu_native"]["anees"],
                c=row["sciu_native"]["complement_anees"],
                np=row["sciu_native"]["private_rmse"],
                cp=row["cu"]["private_rmse"],
            )
        )
    lines.extend(
        [
            "",
            f"- native ANEES <= 1.15: {a6['native_anees_le_1_15']}",
            f"- native complement NEES <= 1.15: {a6['native_complement_anees_le_1_15']}",
            f"- native logdet < CU: {a6['native_logdet_lt_cu']}",
            f"- native private RMSE < CU: {a6['native_private_rmse_lt_cu']}",
            "",
            "## B. $I_o$ sensitivity (restriction SCIU)",
            "",
            "| cell | true | wrong | extra | empty |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in CELLS:
        row = payload["io"]["means"][cell]
        lines.append(
            "| {cell} | {t:.3f} | {w:.3f} | {e:.3f} | {m:.3f} |".format(
                cell=cell,
                t=row["true"]["anees"],
                w=row["wrong"]["anees"],
                e=row["extra"]["anees"],
                m=row["empty"]["anees"],
            )
        )
    lines.extend(
        [
            "",
            f"- true holds 1.15 gate: {io['true_anees_le_1_15']}",
            f"- empty fails conflict-only ({io['empty_conflict_anees']:.3f} > 1.20): {io['empty_fails_conflict']}",
            f"- wrong fails conflict-only ({io['wrong_conflict_anees']:.3f} > 1.20): {io['wrong_fails_conflict']}",
            f"- extra private tax corr-only ({io['extra_private_rmse_corr_only']:.3f} > {io['true_private_rmse_corr_only']:.3f}): {io['extra_pays_private_tax']}",
            "",
            "Coupled-$P$ probe is a stress of remainder A6, not a retune of the frozen 2x2 gate.",
            "Path A SRCF is not the object. GPU unauthorized.",
        ]
    )
    (out_dir / "GATE_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "results" / "a6_io"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    a6_rows, a6_means, a6_gate = evaluate_a6(args.seeds, args.n)
    io_rows, io_means, io_gate = evaluate_io(args.seeds, args.n)
    coupled_rows, coupled_means = evaluate_coupled(args.seeds, min(args.n, 2000), rho=0.45)
    payload = {
        "object": "sciu_a6_io",
        "path": "B",
        "gpu_authorized": False,
        "n_per_cell": int(args.n),
        "seeds": [int(s) for s in args.seeds],
        "a6": {"seed_rows": a6_rows, "means": a6_means, "gate": a6_gate},
        "io": {"seed_rows": io_rows, "means": io_means, "gate": io_gate},
        "coupled": {"rho": 0.45, "n": min(args.n, 2000), "seed_rows": coupled_rows, "means": coupled_means},
    }
    (out_dir / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload, out_dir)
    print(
        json.dumps(
            {
                "a6_passed": a6_gate["passed"],
                "io_passed": io_gate["passed"],
                "native_anees": a6_gate["native_anees"],
                "empty_conflict": io_gate["empty_conflict_anees"],
                "wrong_conflict": io_gate["wrong_conflict_anees"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

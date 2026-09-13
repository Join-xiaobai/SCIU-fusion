# SCIU-fusion

CPU reference for **Split Covariance Intersection--Union (SCIU)**.

Two packets can fail fusion in two structurally different ways at once:
unknown cross-correlation of the same object, and an exclusive hypothesis on
an overlapping coordinate. SCIU covers that product set without a detector:

```
(u_o, U_o) = CU(μ_A[I_o], P_A[I_o], μ_B[I_o], P_B[I_o])
(μ_⊥, P_⊥) = SCI(complement Joseph split)
P_SCIU = blkdiag(U_o, P_⊥)
```

Empty overlap recovers SCI. Empty complement recovers CU. With a private
complement, SCIU is not full-state CU. Joint Loewner domination of the full
mean-square error is **not** claimed.

## What this repository is

- NumPy operators: SCIU, SCI, CU, CI, ICI, and the sequential CI-family wrapper
- Algebraic identities (empty overlap / empty complement / Prop. 7 CU tax)
- CPU known-truth generators for the frozen 2×2, A6 / `I_o`, and sequential m=3 gates
- Recorded aggregate gate reports only (no simulated rows, no patient rows)

## What this repository is not

- Not Sequential Reliability-Conflict Fusion (Path A fallback)
- Not a residual switch between CI and CU
- Not a MIMIC / eICU / DenseNet training tree
- Not a claim that SCIU jointly Loewner-dominates the MSE

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ and NumPy 1.24+ are enough. No GPU.

## Tests

From the repository root:

```bash
python tests/test_sciu_identities.py
python tests/test_sciu_loewner_marginal.py
python tests/test_sciu_native_a6.py
python tests/test_sciu_prop7_tax.py
```

## Reproduce the CPU gates

Defaults write under `results/`. They do **not** overwrite the recorded reports
unless you pass `--out` to the same directory.

```bash
# 2x2 known-truth (n=4000, seeds 17 29 43)
python scripts/run_sciu_2x2_pilot.py --out results/2x2_rerun

# A6 native complement and I_o sensitivity
python scripts/run_sciu_a6_io.py --out results/a6_io_rerun

# sequential three-source support (order A-B-C; not order-invariant)
python scripts/run_sciu_m3.py --out results/m3_rerun
```

Recorded summaries:

- `results/2x2/GATE_REPORT.md`
- `results/a6_io/GATE_REPORT.md`
- `results/m3/GATE_REPORT.md`

## Operator

The public object is `sciu_fuse_pair` / `batch_sciu_fuse_pair` in
`src/models/srcf_filter.py`. Native-block Joseph (A6) is
`sciu_native_fuse_pair`. Comparators live in the same module so the identities
can be checked against CI / SCI / ICI / CU on the same packets.

## Licence

MIT. See `LICENSE`.

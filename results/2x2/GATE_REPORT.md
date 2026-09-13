# SCIU 2x2 known-truth gate

n=4000 per cell; seeds=[17, 29, 43]; CPU only.
passed=True

| cell | kalman ANEES | CI | SCI | ICI | CU | SCIU | SRCF | SCIU logdet | CU logdet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1.012 | 0.584 | 1.002 | 0.864 | 0.367 | 0.806 | 0.692 | -6.980 | -1.390 |
| corr_only | 1.198 | 0.669 | 1.148 | 0.917 | 0.481 | 0.913 | 0.697 | -6.354 | -0.758 |
| conflict_only | 5.721 | 2.139 | 5.591 | 1.361 | 0.617 | 0.976 | 0.875 | -5.456 | -0.019 |
| both | 3.657 | 1.781 | 3.417 | 1.805 | 0.605 | 0.993 | 0.850 | -5.250 | 0.222 |

- SCIU ANEES <= 1.15 every cell: True
- Kalman overlap ANEES > 1.20 corr-only: True
- CI or SCI ANEES > 1.20 conflict-only: True
- SCIU logdet < CU on corr-only and both: True
- SCIU private RMSE < CU on corr-only and both: True

Path A SRCF is recorded, not the Path B object. v6 DGP was not retuned.

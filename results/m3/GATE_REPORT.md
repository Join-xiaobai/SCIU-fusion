# Sequential three-source SCIU support

n=2000; seeds=[17, 29, 43]; order A-B-C; CPU only.
passed=True

| cell | CI ANEES | SCI | CU | SCIU | SCIU priv | CU priv |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.451 | 0.859 | 0.400 | 0.711 | 0.300 | 0.812 |
| corr_only | 0.548 | 1.032 | 0.503 | 0.805 | 0.298 | 0.816 |
| conflict_only | 1.082 | 3.026 | 0.450 | 0.794 | 0.297 | 0.781 |
| both | 0.921 | 2.137 | 0.486 | 0.836 | 0.299 | 0.784 |

- SCIU ANEES <= 1.15: True
- CI or SCI conflict > 1.20: True
- SCIU private RMSE < CU: True

Not order-invariant. Frozen two-source 2x2 was not overwritten.

# SCIU A6 native complement and I_o sensitivity

n=4000; seeds=[17, 29, 43]; CPU only.
a6_passed=True
io_passed=True

## A. Native complement Joseph (A6 revision)

| cell | restriction ANEES | native ANEES | native complement NEES | native priv RMSE | CU priv RMSE |
|---|---:|---:|---:|---:|---:|
| none | 0.806 | 0.806 | 0.997 | 0.296 | 0.673 |
| corr_only | 0.913 | 0.913 | 0.997 | 0.296 | 0.682 |
| conflict_only | 0.976 | 0.976 | 0.986 | 0.294 | 0.619 |
| both | 0.993 | 0.993 | 0.994 | 0.295 | 0.624 |

- native ANEES <= 1.15: True
- native complement NEES <= 1.15: True
- native logdet < CU: True
- native private RMSE < CU: True

## B. $I_o$ sensitivity (restriction SCIU)

| cell | true | wrong | extra | empty |
|---|---:|---:|---:|---:|
| none | 0.806 | 0.837 | 0.584 | 1.002 |
| corr_only | 0.913 | 0.985 | 0.703 | 1.148 |
| conflict_only | 0.976 | 5.563 | 0.687 | 5.591 |
| both | 0.993 | 3.528 | 0.712 | 3.417 |

- true holds 1.15 gate: True
- empty fails conflict-only (5.591 > 1.20): True
- wrong fails conflict-only (5.563 > 1.20): True
- extra private tax corr-only (0.577 > 0.296): True

Coupled-$P$ probe is a stress of remainder A6, not a retune of the frozen 2x2 gate.
Path A SRCF is not the object. GPU unauthorized.
